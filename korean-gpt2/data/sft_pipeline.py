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
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from .tokenizer import get_tokenizer

class InstructionDataset(IterableDataset):
    """
    Formats instruction/response pairs for SFT in streaming mode.
    Only the loss on the assistant response part is computed.
    """
    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 1024,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.eos_token_id

        self.user_token = "<|user|>\n"
        self.thought_token = "\n<|thought|>\n"
        self.assistant_token = "\n<|assistant|>\n"

    def __iter__(self):
        for row in self.dataset:
            # Handle standard formats
            instruction = row.get("instruction") or row.get("question", "")
            input_text = row.get("input", "")
            output_text = row.get("output") or row.get("response", "")
            thought = row.get("thought") or row.get("reasoning", "")
            
            # Handle heegyu/open-korean-instructions format (<usr>...<bot>...)
            text_field = row.get("text", "")
            if "<usr>" in text_field and "<bot>" in text_field:
                try:
                    parts = text_field.split("<bot>")
                    prompt = parts[0].replace("<usr>", "").strip()
                    output_text = parts[1].strip()
                    thought = "" # No thought in this dataset
                except:
                    continue
            else:
                if input_text and str(input_text).strip():
                    prompt = f"{instruction}\n\n{input_text}"
                else:
                    prompt = instruction

            if not prompt or not output_text:
                continue

            if thought:
                full_text = f"{self.user_token}{prompt}{self.thought_token}{thought}{self.assistant_token}{output_text}"
            else:
                full_text = f"{self.user_token}{prompt}{self.assistant_token}{output_text}"
            
            # Optimization: Character-level early exit to avoid tokenizing massive texts
            # Average 2-3 chars per token for Korean. If > 5000 chars, it's likely > 2000 tokens.
            if len(full_text) > 5000:
                continue

            # Tokenize prompt first to get correct mask length
            prompt_marker = f"{self.user_token}{prompt}"
            if thought:
                 prompt_marker += f"{self.thought_token}{thought}"
            prompt_marker += self.assistant_token
            
            prompt_ids = self.tokenizer.encode(prompt_marker, truncation=True, max_length=self.seq_len)
            prompt_len = len(prompt_ids)

            # Optimization: Skip if prompt alone consumes most of the context
            if prompt_len >= self.seq_len - 10: 
                continue

            # Tokenize full text with efficient truncation
            token_ids = self.tokenizer.encode(full_text, truncation=True, max_length=self.seq_len)
            token_ids.append(self.eos_id)
            if len(token_ids) > self.seq_len:
                token_ids = token_ids[:self.seq_len]

            pad_len = self.seq_len - len(token_ids)
            padded_tokens = token_ids + [self.eos_id] * pad_len

            x = torch.tensor(padded_tokens[:-1], dtype=torch.long)
            y = torch.tensor(padded_tokens[1:], dtype=torch.long)

            mask = torch.ones_like(y)
            mask[:prompt_len - 1] = 0
            if pad_len > 0:
                mask[-pad_len:] = 0
                
            y = y.masked_fill(mask == 0, -1)
            
            # Final check before yielding: do we actually have something to train on?
            if (y != -1).any():
                yield {"input_ids": x, "labels": y}

class SFTPipeline:
    def __init__(
        self,
        dataset_name: str = "heegyu/open-korean-instructions",
        tokenizer: PreTrainedTokenizer = None,
        seq_len: int = 512,
        max_samples: int = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer or get_tokenizer()
        self.seq_len = seq_len
        self.max_samples = max_samples

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        print(f"🚀 Loading SFT dataset (Streaming): {self.dataset_name}")
        ds = load_dataset(self.dataset_name, split="train", streaming=True)
        
        if shuffle:
            ds = ds.shuffle(seed=42, buffer_size=10)
            
        if self.max_samples:
            ds = ds.take(self.max_samples)

        dataset = InstructionDataset(
            dataset=ds,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
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
