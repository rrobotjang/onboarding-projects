"""
Tokenizer utilities for Korean GPT-2.

Uses the KoGPT2 tokenizer (skt/kogpt2-base-v2) from Hugging Face,
which is a BPE tokenizer pretrained on Korean text with 51200 vocab size.
"""

from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(
    tokenizer_name: str = "skt/kogpt2-base-v2",
) -> PreTrainedTokenizer:
    """
    Load a pretrained Korean BPE tokenizer.

    Args:
        tokenizer_name: Hugging Face tokenizer ID. Defaults to the
                        SKT KoGPT2 tokenizer which is well-suited for Korean.

    Returns:
        A PreTrainedTokenizer instance ready for encoding Korean text.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Add special tokens for chat/SFT
    special_tokens = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|thought|>"]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Ensure we have a pad token (KoGPT2 may not set one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tok = get_tokenizer()
    print(f"Base vocab size: {tok.vocab_size}")
    print(f"Total length:    {len(tok)}")
    print(f"EOS token:       {tok.eos_token} (id={tok.eos_token_id})")
    print(f"PAD token:       {tok.pad_token} (id={tok.pad_token_id})")
    print(f"User token ID:   {tok.convert_tokens_to_ids('<|user|>')}")
    print(f"Assistant ID:    {tok.convert_tokens_to_ids('<|assistant|>')}")
    print(f"Thought ID:      {tok.convert_tokens_to_ids('<|thought|>')}")

    sample = "안녕하세요, 저는 GPT-2 모델입니다. 한국어를 학습합니다."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"\nOriginal:  {sample}")
    print(f"Token IDs: {encoded}")
    print(f"Decoded:   {decoded}")
    print("✅ Tokenizer test passed!")
