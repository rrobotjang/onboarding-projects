"""
Training script for Korean GPT-2.

Usage:
    python3 train.py                              # Train on all sources
    python3 train.py --sources wikipedia           # Train on Wikipedia only
    python3 train.py --n_layers 6 --d_model 512   # Smaller model for testing
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from model.gpt2 import GPT2, GPT2Config
from data.pipeline import KoreanDatasetPipeline
from data.tokenizer import get_tokenizer


def get_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    device = get_device()
    print(f"🖥️  Device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"📝 Vocab size: {vocab_size}")

    # Data pipeline
    sources = args.sources.split(",") if args.sources else None
    pipeline = KoreanDatasetPipeline(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_youtube_samples=args.max_youtube_samples,
        sources=sources,
    )
    dataset = pipeline.build_dataset()

    if len(dataset) == 0:
        print("❌ No data chunks created. Check your data sources.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    config = GPT2Config(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = GPT2(config).to(device)

    # ------------------------------------------------------------------ #
    # Optimizer (AdamW with GPT-2-style settings)
    # ------------------------------------------------------------------ #
    # Separate weight decay groups: no decay for biases and LayerNorm
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Cosine learning rate scheduler with warmup
    total_steps = args.epochs * len(dataloader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🏋️ Starting training")
    print(f"   Epochs:        {args.epochs}")
    print(f"   Batch size:    {args.batch_size}")
    print(f"   Seq length:    {args.seq_len}")
    print(f"   Total steps:   {total_steps}")
    print(f"   Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            logits, loss = model(input_ids, targets=labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if (batch_idx + 1) % args.log_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                tokens_per_sec = (
                    (batch_idx + 1) * args.batch_size * args.seq_len / elapsed
                )
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:,.0f}"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        print(
            f"\n📊 Epoch {epoch+1} complete | "
            f"Avg loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": best_loss,
                },
                ckpt_path,
            )
            print(f"💾 Saved best checkpoint to {ckpt_path}")

        # Generate a sample at the end of each epoch
        if args.generate_every_epoch:
            model.eval()
            prompt_text = "한국의"  # "Korea's"
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids, max_new_tokens=64, temperature=0.8, top_k=50
                )
            sample_text = tokenizer.decode(generated[0].tolist())
            print(f"📝 Sample generation: {sample_text[:300]}")
            model.train()

        print()

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "config": config,
            "loss": avg_epoch_loss,
        },
        final_path,
    )
    print(f"\n🏁 Training complete! Final model saved to {final_path}")
    print(f"   Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Korean GPT-2")

    # Data
    parser.add_argument(
        "--sources", type=str, default=None,
        help="Comma-separated list of sources: wikipedia,webtext,historical,youtube"
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_youtube_samples", type=int, default=50000)

    # Model
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Logging & checkpoints
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--generate_every_epoch", action="store_true", default=True)

    args = parser.parse_args()

    # Derived
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model

    train(args)


if __name__ == "__main__":
    main()
