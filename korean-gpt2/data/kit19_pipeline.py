import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from transformers import PreTrainedTokenizer
import json
from pathlib import Path
import random

from data.tokenizer import get_tokenizer

TASK_INSTRUCTIONS = {
    "hatespeech_detection": "다음 문장이 혐오 표현인지 판별해 주세요.",
    "sentiment_analysis": "다음 문장의 감정을 분석해 주세요.",
    "topic_classification": "다음 글의 주제를 분류해 주세요.",
    "natural_language_inference": "두 문장 사이의 논리적 관계(함의, 모순, 중립)를 판별해 주세요.",
    "named_entity_recognition": "다음 문장에서 주요 개체명(인물, 장소, 조직 등)을 추출해 주세요.",
    "summarization": "다음 글을 핵심 내용 위주로 요약해 주세요.",
    "question_answering": "주어진 질문에 대해 정확한 답변을 작성해 주세요.",
    "machine_reading_comprehension": "주어진 지문을 읽고 질문에 답해 주세요.",
    "sarcasm_detection": "다음 문장에 비꼬는 표현이 포함되어 있는지 확인해 주세요.",
    "paraphrase_detection": "두 문장이 동일한 의미를 가지고 있는지 판별해 주세요.",
    "semantic_textual_similarity": "두 문장의 의미적 유사도를 측정해 주세요.",
    "irony_detection": "다음 문장이 반어법인지 판별해 주세요.",
    "dialect_classification": "다음 문장이 어느 지역 방언인지 분류해 주세요.",
    "styled_dialogue_generation": "주어진 상황에 맞는 적절한 답변을 생성해 주세요.",
    "politeness_classification": "다음 문장의 높임말 사용 여부 및 공손함을 판별해 주세요.",
    "translation_ko_en": "다음 문장을 영어로 번역해 주세요.",
    "grammar_correction": "다음 문장의 문법 오류를 수정해 주세요.",
    "intent_classification": "사용자의 의도를 파악해 주세요.",
    "news_headline_generation": "다음 뉴스 기사에 어울리는 헤드라인을 생성해 주세요."
}

class KIT19Dataset(IterableDataset):
    """
    Handles multi-task instruction pairs from KIT-19 in streaming mode.
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
        self.assistant_token = "\n<|assistant|>\n"

    def __iter__(self):
        for row in self.dataset:
            instruction = row.get("instruction")
            task = row.get("task", "general")
            
            # Map instruction if None
            if instruction is None or not instruction.strip():
                instruction = TASK_INSTRUCTIONS.get(task, "다음 지시에 따라 답해 주세요.")
            
            input_text = row.get("input", "").strip()
            output = row.get("output", "").strip()
            
            if not output:
                continue

            # Format: <|user|>\n[Instruction]\n[Input]\n<|assistant|>\n[Output]<|endoftext|>
            prompt = f"{instruction}\n{input_text}".strip()
            full_text = f"{self.user_token}{prompt}{self.assistant_token}{output}"
            
            token_ids = self.tokenizer.encode(full_text)
            token_ids.append(self.eos_id)

            if len(token_ids) > self.seq_len:
                token_ids = token_ids[:self.seq_len]

            # Mask the prompt for loss calculation
            prompt_marker = f"{self.user_token}{prompt}{self.assistant_token}"
            prompt_ids = self.tokenizer.encode(prompt_marker)
            prompt_len = len(prompt_ids)

            # Padding & Tensorizing
            pad_len = self.seq_len - len(token_ids)
            padded_tokens = token_ids + [self.eos_id] * pad_len

            x = torch.tensor(padded_tokens[:-1], dtype=torch.long)
            y = torch.tensor(padded_tokens[1:], dtype=torch.long)

            # Mask loss on prompt and padding
            mask = torch.ones_like(y)
            mask[:prompt_len - 1] = 0
            if pad_len > 0:
                mask[-pad_len:] = 0
                
            y = y.masked_fill(mask == 0, -1)

            yield {"input_ids": x, "labels": y}

class KIT19Pipeline:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        seq_len: int = 1024,
        max_samples: int = None,
    ):
        self.tokenizer = tokenizer or get_tokenizer()
        self.seq_len = seq_len
        self.max_samples = max_samples

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        print(f"🚀 Loading KIT-19 dataset (Streaming)...")
        repo_name = "snunlp/KIT-19-ToolKit-100000"
        
        try:
            ds = load_dataset(repo_name, split="train", streaming=True)
            if shuffle:
                ds = ds.shuffle(seed=42, buffer_size=5000)
        except Exception as e:
            print(f"⚠️ Could not load {repo_name} ({e}). Using mock fallback.")
            mock_path = Path(__file__).parent / "mock_kit19.json"
            with open(mock_path, "r", encoding="utf-8") as f:
                mock_data = json.load(f)
            from datasets import Dataset as HFDataset
            ds = HFDataset.from_list(mock_data)
        
        if self.max_samples:
            ds = ds.take(self.max_samples) if hasattr(ds, "take") else ds.select(range(min(len(ds), self.max_samples)))

        dataset = KIT19Dataset(ds, self.tokenizer, self.seq_len)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False, # MPS doesn't like pin_memory
        )

if __name__ == "__main__":
    print("Testing KIT-19 Pipeline (Streaming)...")
    pipeline = KIT19Pipeline(max_samples=5)
    dl = pipeline.get_dataloader(batch_size=2)
    batch = next(iter(dl))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print("✅ KIT-19 Pipeline test passed!")
