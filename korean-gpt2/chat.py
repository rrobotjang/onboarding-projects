"""
Interactive Chat Interface for Fine-Tuned Korean GPT-2.

Usage:
    python3 chat.py --model checkpoints/sft/chat_model_best.pt
"""

import argparse
import sys

import torch
from model.gpt2 import GPT2, GPT2Config
from data.tokenizer import get_tokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device):
    print(f"📥 Loading chat model from {checkpoint_path}...")
    tokenizer = get_tokenizer()
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
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
        model.eval()
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def chat_loop(model: GPT2, tokenizer, device: torch.device, args):
    print("\n" + "="*50)
    print("🤖 Korean GPT-2 Chatbot")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")

    user_token = "<|user|>\n"
    thought_token = "\n<|thought|>\n"
    assistant_token = "\n<|assistant|>\n"
    end_token = "<|endoftext|>"
    
    # Simple history management for multi-turn
    history = ""
    
    while True:
        try:
            user_input = input("You 🧑: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue

        # Append new user message to history
        # We prompt the model to generate the thought first
        history += f"{user_token}{user_input}{thought_token}"
        
        input_ids = tokenizer.encode(history, return_tensors="pt").to(device)

        # Truncate history if it gets too long
        max_context = model.config.max_seq_len - args.max_new_tokens
        if input_ids.size(1) > max_context:
            print("⚠️ Context limit reached, clearing history...")
            history = f"{user_token}{user_input}{assistant_token}"
            input_ids = tokenizer.encode(history, return_tensors="pt").to(device)

        print("Bot 🤖: ", end="", flush=True)

        with torch.no_grad():
            generated_ids = model.generate(
                idx=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )

        # Extract only the newly generated tokens
        new_tokens_idx = input_ids.size(1)
        response_ids = generated_ids[0, new_tokens_idx:]
        
        # Stop at EOS <|endoftext|>
        eos_id = tokenizer.eos_token_id
        if eos_id in response_ids:
            stop_idx = (response_ids == eos_id).nonzero(as_tuple=True)[0][0]
            response_ids = response_ids[:stop_idx]

        response_text = tokenizer.decode(response_ids.tolist())
        
        # Split into thought and assistant parts if both exist
        if assistant_token in response_text:
            parts = response_text.split(assistant_token)
            thought_part = parts[0].strip()
            assistant_part = parts[1].strip()
            
            # Print thought in a different style if needed, or just print it
            if thought_part:
                print(f"\n[Thinking]\n{thought_part}\n")
            print(f"Bot 🤖: {assistant_part}")
        else:
            # If model didn't generate assistant token yet, just print what we have
            print(response_text)
        
        # Add generated answer to history
        history += response_text + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SFT model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    args = parser.parse_args()
    
    device = get_device()
    model, tokenizer = load_model(args.model, device)
    
    chat_loop(model, tokenizer, device, args)
