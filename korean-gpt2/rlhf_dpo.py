"""
Stage 3 Training Script: Direct Preference Optimization (DPO).
Aligns the SFT model with human preferences without a separate reward model.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

from model.gpt2 import GPT2, GPT2Config
from data.dpo_pipeline import DPOPipeline
from data.tokenizer import get_tokenizer

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    DPO Loss calculation.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    
    return losses.mean(), rewards

def get_batch_logps(logits, labels):
    """
    Compute log probabilities of labels under model logits.
    """
    # Shift logits and labels for causal LM
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    
    # Filter labels to keep only non-masked ones (-100 is cross entropy ignore index)
    mask = (labels != -100)
    
    # (batch, seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs for the actual labels
    # labels[mask] will pull out the labels we care about
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2).masked_fill(labels == -100, 0)).squeeze(2)
    
    return (per_token_logps * mask).sum(-1)

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    tokenizer = get_tokenizer()
    pipeline = DPOPipeline(tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataloader = pipeline.get_dataloader(batch_size=args.batch_size)

    # Load Policy Model (The SFT-tuned model)
    print(f"📥 Loading SFT policy model weights from {args.sft_model}...")
    checkpoint = torch.load(args.sft_model, map_location="cpu", weights_only=False)
    
    # Force 125M small config
    config = GPT2Config(vocab_size=len(tokenizer))
    config.use_checkpointing = args.use_checkpointing
    
    policy_model = GPT2(config)
    try:
        policy_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("✅ Weights loaded into policy model.")
    except Exception as e:
        print(f"⚠️ Weight load failed ({e}). Starting with random initialization...")

    policy_model.to(device)
    
    # 2. Reference Model (Freeze it)
    print("❄️ Creating reference model (copy of policy)...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting DPO Training...")
    for epoch in range(args.epochs):
        policy_model.train()
        for batch_idx, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # Policy forward pass
            logits_c, _ = policy_model(chosen_ids)
            logits_r, _ = policy_model(rejected_ids)
            
            policy_chosen_logps = get_batch_logps(logits_c, chosen_labels)
            policy_rejected_logps = get_batch_logps(logits_r, rejected_labels)

            # Reference forward pass (no grads)
            with torch.no_grad():
                ref_logits_c, _ = ref_model(chosen_ids)
                ref_logits_r, _ = ref_model(rejected_ids)
                
                ref_chosen_logps = get_batch_logps(ref_logits_c, chosen_labels)
                ref_rejected_logps = get_batch_logps(ref_logits_r, rejected_labels)

            loss, rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % args.log_every == 0:
                print(f"Step {batch_idx+1} | Loss: {loss.item():.4f} | Reward: {rewards.mean().item():.4f}")

            if (batch_idx + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"dpo_model_step_{batch_idx+1}.pt")
                torch.save({"model_state_dict": policy_model.state_dict(), "config": config}, ckpt_path)

    # Final Save
    final_path = os.path.join(args.output_dir, "dpo_model_final.pt")
    torch.save({"model_state_dict": policy_model.state_dict(), "config": config}, final_path)
    print(f"🏁 DPO Alignment Complete! Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage3_dpo")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6) # Lower for DPO
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--use_checkpointing", action="store_true", default=True)

    args = parser.parse_args()
    train(args)
