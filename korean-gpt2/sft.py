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
    
    # We explicitly want the new 125M config (Small)
    config = GPT2Config(vocab_size=len(tokenizer))
    config.use_checkpointing = args.use_checkpointing
    
    model = GPT2(config)
    
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Weights loaded from checkpoint (where possible).")
    except Exception as e:
        print(f"⚠️ Weight load failed ({e}). Starting with random initialization...")
        
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
    # Optimizer (Lower Learning Rate and Stable Epsilon for SFT)
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        eps=1e-5 # Optimization: Increase eps for training stability on MPS
    )
    
    # In streaming mode, we don't know total_steps easily, so we use a simpler scheduler or fixed warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad()

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"💬 Starting Supervised Fine-Tuning SFT (Streaming)")
    print(f"   Batch size:    {args.batch_size}")
    print(f"   Seq length:    {args.seq_len}")
    print(f"   Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    global_step = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, targets=labels)
            
            # Optimization: Skip if loss is NaN (data or numerical issue)
            if torch.isnan(loss):
                print(f"⚠️ Warning: NaN loss at batch {batch_idx+1}, skipping...", flush=True)
                optimizer.zero_grad()
                continue

            loss = loss / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * args.grad_accum_steps

            if (batch_idx + 1) % 10 == 0:
                 print(f"   [Micro-batch {batch_idx+1}/{args.grad_accum_steps}] Loss: {loss.item()*args.grad_accum_steps:.4f}", flush=True)

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    current_loss = running_loss / args.log_every
                    print(f"Epoch {epoch+1} | Step {global_step} | Loss: {current_loss:.4f}", flush=True)
                    running_loss = 0.0

            # Step-based periodic saving and testing
            if global_step > 0 and global_step % args.save_every == 0 and (batch_idx + 1) % args.grad_accum_steps == 0:
                # Save Latest
                latest_path = os.path.join(args.output_dir, "latest_sft.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                    },
                    latest_path,
                )
                
                # Test Generation
                model.eval()
                print(f"\n� Step {global_step} | Running Chat Test...", flush=True)
                prompt = "<|user|>\n한국의 수도는 어디인가요?\n<|thought|>\n"
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated = model.generate(prompt_ids, max_new_tokens=64, temperature=0.7)
                print(f"🤖 Chat Test:\n{tokenizer.decode(generated[0].tolist())}\n", flush=True)
                model.train()

        print(f"\n📊 Epoch {epoch+1} complete | Time: {time.time() - epoch_start:.1f}s", flush=True)

    print("\n🏁 SFT Training complete!", flush=True)

    print("\n🏁 SFT Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--dataset", type=str, default="heegyu/open-korean-instructions")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--use_checkpointing", action="store_true", default=True)

    args = parser.parse_args()
    train(args)
