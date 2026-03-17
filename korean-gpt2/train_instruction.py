"""
Stage 1 Training Script: Instruction Tuning with KIT-19.
This script focuses on broad task capability (NLP tasks) before conversational SFT.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from model.gpt2 import GPT2, GPT2Config
from data.kit19_pipeline import KIT19Pipeline
from data.tokenizer import get_tokenizer

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    tokenizer = get_tokenizer()
    
    # Pipeline
    pipeline = KIT19Pipeline(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples
    )
    dataloader = pipeline.get_dataloader(batch_size=args.batch_size)

    # Load Base Model (The 1.3B model we configured)
    print(f"📥 Loading base model weights from {args.base_model}...")
    checkpoint = torch.load(args.base_model, map_location="cpu", weights_only=False)
    
    # We explicitly want the new 1.3B config
    config = GPT2Config(vocab_size=len(tokenizer))
    config.use_checkpointing = args.use_checkpointing
    
    model = GPT2(config)
    
    try:
        # Try loading weights. If it fails (config mismatch), it will start fresh.
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Weights loaded from checkpoint (where possible).")
    except Exception as e:
        print(f"⚠️ Weight mismatch/load failure ({e}). Starting with random initialization for 1.3B config...")
        
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready. Global Config: {config.n_layers} layers, {config.d_model} hidden.")
    print(f"✅ Params: {total_params/1e6:.2f}M ({total_params/1e9:.2f}B)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, targets=labels)
            
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_every == 0:
                print(f"Epoch {epoch+1} | Step {batch_idx+1} | Loss: {loss.item()*args.grad_accum_steps:.4f}")

            if (batch_idx + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"kit19_model_step_{batch_idx+1}.pt")
                torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}")

    # Final Save
    final_path = os.path.join(args.output_dir, "kit19_model_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, final_path)
    print(f"🏁 Stage 1 complete! Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage1_kit19")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=6e-4) # Standard for Small
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--use_checkpointing", action="store_true", default=True)

    args = parser.parse_args()
    train(args)
