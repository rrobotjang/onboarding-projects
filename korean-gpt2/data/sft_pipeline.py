"""
Supervised Fine-Tuning (SFT) Data Pipeline for Korean GPT-2.

This pipeline loads an instruction dataset (e.g., devngho/korean-instruction-mix)
and formats it into conversational prompts for the chatbot to learn from.

Format used:
<|user|>
[Question]
<|thought|>
[Reasoning steps]
<|assistant|>
[Answer]<|endoftext|>
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from .tokenizer import get_tokenizer


class InstructionDataset(Dataset):
    """
    Formats instruction/response pairs for SFT.
    Only the loss on the <|assistant|> response part is computed during training.
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.eos_token_id

        self.user_token = "<|user|>\n"
        self.thought_token = "\n<|thought|>\n"
        self.assistant_token = "\n<|assistant|>\n"

        print(f"🔤 Formatting {len(dataset)} instruction pairs...")
        self.examples = []

        for i, row in enumerate(dataset):
            # Support different column naming conventions
            instruction = row.get("instruction") or row.get("question", "")
            input_text = row.get("input", "")
            output_text = row.get("output") or row.get("response", "")
            thought = row.get("thought") or row.get("reasoning", "") # For Reasoning datasets

            # Combine instruction and input if both exist
            if input_text and str(input_text).strip():
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction

            if not prompt or not output_text:
                continue

            # Format: <|user|>\n [Prompt] \n<|thought|>\n [Reasoning] \n<|assistant|>\n [Completion] <|endoftext|>
            if thought:
                full_text = f"{self.user_token}{prompt}{self.thought_token}{thought}{self.assistant_token}{output_text}"
            else:
                # Fallback to simple format if no thought
                full_text = f"{self.user_token}{prompt}{self.assistant_token}{output_text}"
            
            # Encode
            token_ids = tokenizer.encode(full_text)
            token_ids.append(self.eos_id)

            # We need to compute loss ONLY on the assistant's output + thought, not the user's prompt.
            # Find where the prompt ends
            prompt_marker = f"{self.user_token}{prompt}"
            prompt_ids = tokenizer.encode(prompt_marker)
            prompt_len = len(prompt_ids)

            # Truncate if too long (leave room for at least 1 response token)
            if len(token_ids) > self.seq_len:
                if prompt_len >= self.seq_len - 1:
                    continue  # Prompt is too long to fit, skip this example
                token_ids = token_ids[:self.seq_len]

            self.examples.append({
                "input_ids": token_ids,
                "prompt_len": prompt_len,
            })

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} examples...")

        print(f"✅ Filtered to {len(self.examples)} valid examples.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        tokens = example["input_ids"]
        prompt_len = example["prompt_len"]

        # Pad sequence to max length
        pad_len = self.seq_len - len(tokens)
        pad_id = self.eos_id  # Use EOS as PAD if needed
        padded_tokens = tokens + [pad_id] * pad_len

        x = torch.tensor(padded_tokens[:-1], dtype=torch.long)
        y = torch.tensor(padded_tokens[1:], dtype=torch.long)

        # Mask out the prompt and padding from the loss calculation
        # PyTorch cross_entropy ignores index -1
        mask = torch.ones_like(y)
        # 1. Mask the prompt
        mask[:prompt_len - 1] = 0
        # 2. Mask the padding
        if pad_len > 0:
            mask[-pad_len:] = 0
            
        y = y.masked_fill(mask == 0, -1)

        return {"input_ids": x, "labels": y}


class SFTPipeline:
    def __init__(
        self,
        dataset_name: str = "llami-team/Korean-OpenThoughts-114k-Normalized",
        tokenizer: PreTrainedTokenizer = None,
        seq_len: int = 1024,
        max_samples: int = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer or get_tokenizer()
        self.seq_len = seq_len
        self.max_samples = max_samples

    def build_dataset(self) -> InstructionDataset:
        print("=" * 60)
        print(f"🚀 Loading SFT dataset: {self.dataset_name}")
        print("=" * 60)

        # Load from Hugging Face
        ds = load_dataset(self.dataset_name, split="train")
        
        if self.max_samples:
            ds = ds.select(range(min(len(ds), self.max_samples)))

        dataset = InstructionDataset(
            dataset=ds,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
        )

        return dataset

    def get_dataloader(self, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        dataset = self.build_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )


if __name__ == "__main__":
    print("Testing SFT Data Pipeline (1000 samples)...\n")
    pipeline = SFTPipeline(max_samples=1000, seq_len=128)
    ds = pipeline.build_dataset()

    if len(ds) > 0:
        sample = ds[0]
        x = sample["input_ids"]
        y = sample["labels"]

        print(f"\nInput shape: {x.shape}")
        print(f"Label shape: {y.shape}")
        
        tok = pipeline.tokenizer
        print("\nInput Tokens:")
        print(tok.decode(x[x != tok.eos_token_id]))
        
        print("\nTokens trained on (Labels != -1):")
        valid_labels = y[y != -1]
        if len(valid_labels) > 0:
            print(tok.decode(valid_labels))
        else:
            print("Warning: No valid labels found (prompt might be too long)")
    print("\n✅ SFT Pipeline test complete!")
