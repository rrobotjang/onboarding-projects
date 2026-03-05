"""
Korean Data Pipeline for GPT-2 pretraining.

Fetches and combines multiple Korean text sources from Hugging Face:
  1. Korean Wikipedia (devngho/korean_wikipedia)
  2. Korean Web Text (HAERAE-HUB/KOREAN-WEBTEXT)
  3. Korean Historical Corpus (seyoungsong/Open-Korean-Historical-Corpus) — 정약용 era texts
  4. YouTube Commons (PleIAs/YouTube-Commons) — filtered for Korean

All datasets are tokenized, chunked into fixed-length sequences, and served
as a standard PyTorch Dataset for the training loop.
"""

import os
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from transformers import PreTrainedTokenizer

from .tokenizer import get_tokenizer


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_korean_wikipedia(
    split: str = "train",
    streaming: bool = False,
) -> HFDataset:
    """Load Korean Wikipedia articles."""
    print("📚 Loading Korean Wikipedia...")
    ds = load_dataset(
        "devngho/korean_wikipedia",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    # Standardize column name to 'text'
    if "text" not in ds.column_names:
        # Try common alternatives
        for col in ["content", "article", "body"]:
            if col in ds.column_names:
                ds = ds.rename_column(col, "text")
                break
    return ds


def load_korean_webtext(
    split: str = "train",
    streaming: bool = False,
) -> HFDataset:
    """Load HAERAE-HUB Korean web text corpus."""
    print("🌐 Loading Korean Web Text...")
    ds = load_dataset(
        "HAERAE-HUB/KOREAN-WEBTEXT",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    if "text" not in ds.column_names:
        for col in ["content", "document", "body"]:
            if col in ds.column_names:
                ds = ds.rename_column(col, "text")
                break
    return ds


def load_korean_historical_corpus(
    split: str = "train",
    streaming: bool = False,
) -> HFDataset:
    """
    Load Korean historical corpus (old Hangeul and Hanja texts).
    This covers 정약용 (Jeong Yak-yong / Dasan) era classical Korean writings.
    """
    print("📜 Loading Korean Historical Corpus (정약용 era)...")
    ds = load_dataset(
        "seyoungsong/Open-Korean-Historical-Corpus",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    if "text" not in ds.column_names:
        for col in ["content", "sentence", "body", "original"]:
            if col in ds.column_names:
                ds = ds.rename_column(col, "text")
                break
    return ds


def load_youtube_korean_transcripts(
    split: str = "train",
    streaming: bool = True,
    max_samples: int = 50000,
) -> HFDataset:
    """
    Load YouTube Commons and filter for Korean transcripts.
    Streaming is recommended since the full dataset is very large.
    """
    print("🎬 Loading YouTube Korean transcripts...")
    ds = load_dataset(
        "PleIAs/YouTube-Commons",
        split=split,
        streaming=True,  # Always stream — dataset is huge
        trust_remote_code=True,
    )

    # Filter for Korean language
    def is_korean(example):
        lang = example.get("language", example.get("lang", ""))
        return lang in ("ko", "korean", "kor")

    ds = ds.filter(is_korean)

    if not streaming:
        # Take a subset to avoid downloading everything
        samples = []
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            samples.append(example)
        from datasets import Dataset as HFDataset2
        ds = HFDataset2.from_list(samples)

    return ds


# ---------------------------------------------------------------------------
# Tokenized chunked dataset
# ---------------------------------------------------------------------------

class TokenizedTextDataset(Dataset):
    """
    Takes raw text, tokenizes it, concatenates all tokens, and serves
    fixed-length chunks for language modeling.

    This is the standard approach for pretraining: we don't pad or truncate
    individual documents. Instead, we concatenate everything and slice into
    uniform chunks.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 1024,
    ):
        self.seq_len = seq_len

        print(f"🔤 Tokenizing {len(texts)} documents...")
        # Tokenize all texts and concatenate into one big tensor
        all_token_ids = []
        eos_id = tokenizer.eos_token_id

        for i, text in enumerate(texts):
            if text and text.strip():
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(ids)
                all_token_ids.append(eos_id)  # Document separator

            if (i + 1) % 10000 == 0:
                print(f"  Tokenized {i + 1}/{len(texts)} documents...")

        self.data = torch.tensor(all_token_ids, dtype=torch.long)
        self.n_chunks = max(0, (len(self.data) - 1) // self.seq_len)

        print(f"  Total tokens: {len(self.data):,}")
        print(f"  Number of chunks (seq_len={seq_len}): {self.n_chunks:,}")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len

        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]

        return {"input_ids": x, "labels": y}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class KoreanDatasetPipeline:
    """
    End-to-end pipeline that loads, combines, tokenizes, and serves
    Korean datasets for GPT-2 pretraining.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        seq_len: int = 1024,
        max_youtube_samples: int = 50000,
        sources: list[str] | None = None,
    ):
        """
        Args:
            tokenizer: Tokenizer to use. If None, loads the default Korean tokenizer.
            seq_len: Sequence length for training chunks.
            max_youtube_samples: Max samples to pull from YouTube Commons.
            sources: List of sources to include. Options:
                     ["wikipedia", "webtext", "historical", "youtube"]
                     If None, uses all four.
        """
        self.tokenizer = tokenizer or get_tokenizer()
        self.seq_len = seq_len
        self.max_youtube_samples = max_youtube_samples
        self.sources = sources or ["wikipedia", "webtext", "historical", "youtube"]

    def _collect_texts(self) -> list[str]:
        """Fetch all datasets and extract text fields."""
        all_texts = []

        if "wikipedia" in self.sources:
            try:
                ds = load_korean_wikipedia()
                texts = [ex["text"] for ex in ds if ex.get("text")]
                print(f"  Wikipedia: {len(texts):,} documents")
                all_texts.extend(texts)
            except Exception as e:
                print(f"  ⚠️ Wikipedia load failed: {e}")

        if "webtext" in self.sources:
            try:
                ds = load_korean_webtext()
                texts = [ex["text"] for ex in ds if ex.get("text")]
                print(f"  Web text: {len(texts):,} documents")
                all_texts.extend(texts)
            except Exception as e:
                print(f"  ⚠️ Web text load failed: {e}")

        if "historical" in self.sources:
            try:
                ds = load_korean_historical_corpus()
                texts = [ex["text"] for ex in ds if ex.get("text")]
                print(f"  Historical: {len(texts):,} documents")
                all_texts.extend(texts)
            except Exception as e:
                print(f"  ⚠️ Historical corpus load failed: {e}")

        if "youtube" in self.sources:
            try:
                ds = load_youtube_korean_transcripts(
                    streaming=False,
                    max_samples=self.max_youtube_samples,
                )
                texts = [ex["text"] for ex in ds if ex.get("text")]
                print(f"  YouTube: {len(texts):,} documents")
                all_texts.extend(texts)
            except Exception as e:
                print(f"  ⚠️ YouTube load failed: {e}")

        print(f"\n📊 Total documents collected: {len(all_texts):,}")
        return all_texts

    def build_dataset(self) -> TokenizedTextDataset:
        """
        Run the full pipeline: fetch → tokenize → chunk.

        Returns:
            A PyTorch Dataset ready for DataLoader.
        """
        print("=" * 60)
        print("🚀 Building Korean GPT-2 training dataset")
        print("=" * 60)

        texts = self._collect_texts()

        dataset = TokenizedTextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
        )

        print("=" * 60)
        print("✅ Dataset ready!")
        print("=" * 60)

        return dataset

    def get_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Convenience method that returns a ready-to-use DataLoader."""
        dataset = self.build_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing data pipeline with Wikipedia only (smallest source)...\n")

    pipeline = KoreanDatasetPipeline(
        seq_len=256,
        sources=["wikipedia"],  # Just wiki for quick test
    )

    dataset = pipeline.build_dataset()

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample input_ids shape: {sample['input_ids'].shape}")
        print(f"Sample labels shape:    {sample['labels'].shape}")
        print(f"First 20 tokens:        {sample['input_ids'][:20].tolist()}")

        # Decode a snippet
        tok = pipeline.tokenizer
        decoded = tok.decode(sample["input_ids"][:50].tolist())
        print(f"Decoded snippet:        {decoded[:200]}")
    else:
        print("No data chunks created (dataset may be too small).")

    print("\n✅ Pipeline test complete!")
