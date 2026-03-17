"""
DPO (Direct Preference Optimization) Data Pipeline.
Dataset format: {'prompt': str, 'chosen': str, 'rejected': str}
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from data.tokenizer import get_tokenizer

class DPODataset(Dataset):
    """
    Handles preference pairs (chosen, rejected) for DPO.
    """
    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = tokenizer.eos_token_id

        self.user_token = "<|user|>\n"
        self.assistant_token = "\n<|assistant|>\n"

        print(f"📦 Formatting {len(dataset)} DPO pairs...")
        self.pairs = []

        def to_str(val):
            if isinstance(val, list):
                # Handle chat format: [{'role': '...', 'content': '...'}]
                if len(val) > 0 and isinstance(val[0], dict) and "content" in val[0]:
                    return "\n".join([m["content"] for m in val if "content" in m])
                return str(val)
            return str(val).strip()

        for i, row in enumerate(dataset):
            prompt = to_str(row.get("prompt", ""))
            chosen = to_str(row.get("chosen", ""))
            rejected = to_str(row.get("rejected", ""))
            
            if not all([prompt, chosen, rejected]):
                continue

            # Full prompt: <|user|>\n[Prompt]\n<|assistant|>\n
            full_prompt = f"{self.user_token}{prompt}{self.assistant_token}"
            
            # Encode
            prompt_ids = tokenizer.encode(full_prompt)
            chosen_ids = tokenizer.encode(chosen) + [self.eos_id]
            rejected_ids = tokenizer.encode(rejected) + [self.eos_id]

            # Truncate if too long (keeping room for chosen/rejected)
            # DPO requires prompt + response to fit in max_seq_len
            total_chosen = prompt_ids + chosen_ids
            total_rejected = prompt_ids + rejected_ids

            if len(total_chosen) > self.max_seq_len:
                # Truncate chosen, but keep prompt if possible
                chosen_ids = chosen_ids[:max(0, self.max_seq_len - len(prompt_ids))]
                total_chosen = prompt_ids + chosen_ids
            
            if len(total_rejected) > self.max_seq_len:
                rejected_ids = rejected_ids[:max(0, self.max_seq_len - len(prompt_ids))]
                total_rejected = prompt_ids + rejected_ids

            self.pairs.append({
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
            })

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} pairs...")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        
        prompt_ids = pair["prompt_ids"]
        chosen_ids = pair["chosen_ids"]
        rejected_ids = pair["rejected_ids"]

        # Concatenate prompt + chosen, prompt + rejected
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids

        # Loss mask (we only compute loss on the response part)
        chosen_labels = ([-100] * len(prompt_ids)) + chosen_ids
        rejected_labels = ([-100] * len(prompt_ids)) + rejected_ids

        # Pad to max_seq_len
        def pad(ids, labels):
            pad_len = self.max_seq_len - len(ids)
            ids = ids + [self.eos_id] * pad_len
            labels = labels + [-100] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        chosen_ids, chosen_labels = pad(chosen_input_ids, chosen_labels)
        rejected_ids, rejected_labels = pad(rejected_input_ids, rejected_labels)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }

class DPOPipeline:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        max_seq_len: int = 1024,
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",  # Example, can be Korean
    ):
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_seq_len = max_seq_len
        self.dataset_name = dataset_name

    def get_dataloader(self, batch_size: int = 4, split: str = "train_prefs") -> DataLoader:
        print(f"🚀 Loading DPO dataset: {self.dataset_name}...")
        try:
            ds = load_dataset(self.dataset_name, split=split)
        except Exception as e:
            print(f"❌ Could not load {self.dataset_name}: {e}. Using mock data.")
            # Mock binarized dataset
            ds = [
                {"prompt": "안녕하세요?", "chosen": "안녕하세요! 무엇을 도와드릴까요?", "rejected": "네."},
                {"prompt": "오늘 날씨 어때?", "chosen": "오늘 서울은 맑고 따뜻할 예정입니다.", "rejected": "몰라."},
            ]
            
        dataset = DPODataset(ds, self.tokenizer, self.max_seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    pipeline = DPOPipeline()
    dl = pipeline.get_dataloader(batch_size=2)
    batch = next(iter(dl))
    print(f"Batch keys: {batch.keys()}")
    print(f"Chosen shape: {batch['chosen_input_ids'].shape}")
    print("✅ DPO Pipeline test passed!")
