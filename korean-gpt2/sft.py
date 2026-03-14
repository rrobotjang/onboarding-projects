"""
Supervised Fine-Tuning (SFT) script to turn the base GPT-2 into a chatbot.

Usage:
    python3 sft.py --base_model checkpoints/best_model.pt
"""

import argparse
import math
import os
import time

import torch

from model.gpt2 import GPT2, GPT2Config
from data.sft_pipeline import SFTPipeline
from data.tokenizer import get_tokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"🖥️  Device: {device}")

    tokenizer = get_tokenizer()
    
    # ------------------------------------------------------------------ #
    # Load Base Model
    # ------------------------------------------------------------------ #
    print(f"📥 Loading base model from {args.base_model}...")
    checkpoint = torch.load(args.base_model, map_location=device, weights_only=False)
    
    # Checkpoint could be just state dict or full dict
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Fallback to default
        config = GPT2Config(vocab_size=tokenizer.vocab_size)
        
    model = GPT2(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    # Resize embeddings if tokenizer has special tokens
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    print("✅ Base model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Data Pipeline
    # ------------------------------------------------------------------ #
    pipeline = SFTPipeline(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples
    )
    dataloader = pipeline.get_dataloader(batch_size=args.batch_size)

    # ------------------------------------------------------------------ #
    # Optimizer (Lower Learning Rate for SFT)
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * (len(dataloader) // args.grad_accum_steps)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad()

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"💬 Starting Supervised Fine-Tuning SFT")
    print(f"   Epochs:        {args.epochs}")
    print(f"   Batch size:    {args.batch_size}")
    print(f"   Seq length:    {args.seq_len}")
    print(f"   Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, targets=labels)
            
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum_steps

            # Step-based periodic saving (Moved out of epoch end block to step loop)
            if (batch_idx + 1) % args.save_every == 0:
                latest_path = os.path.join(args.output_dir, "latest_sft.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": batch_idx + 1,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "loss": loss.item(),
                    },
                    latest_path,
                )
                print(f"💾 Step {batch_idx+1} | SFT progress saved to {latest_path}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n📊 Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f} | Time: {time.time() - epoch_start:.1f}s")

        # Save Best Checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = os.path.join(args.output_dir, "chat_model_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "loss": best_loss,
                },
                ckpt_path,
            )
            print(f"💾 Saved best SFT checkpoint to {ckpt_path}")

        # Test Generation (Reasoning)
        model.eval()
        prompt = "<|user|>\n지구가 둥근 이유를 단계별로 설명해줘.\n<|thought|>\n"
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(prompt_ids, max_new_tokens=128, temperature=0.7)
        print(f"🤖 Chat Reasoning Test:\n{tokenizer.decode(generated[0].tolist())}\n")

    print("\n🏁 SFT Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--dataset", type=str, default="llami-team/Korean-OpenThoughts-114k-Normalized")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5) # Lower LR for fine-tuning
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500) # Save every 500 steps

    args = parser.parse_args()
    train(args)
