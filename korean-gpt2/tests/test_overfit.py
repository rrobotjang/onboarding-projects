"""
Test that the GPT-2 model can overfit on a tiny sample of Korean text.
This verifies end-to-end correctness of the model + optimizer integration.
"""

import torch
from model.gpt2 import GPT2, GPT2Config
from data.tokenizer import get_tokenizer


def test_overfit():
    print("🧪 Overfit test: training on a single Korean sentence...\n")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Small model for fast test
    config = GPT2Config(
        vocab_size=vocab_size,
        max_seq_len=128,
        n_layers=2,
        n_heads=2,
        d_model=128,
        d_ff=512,
        dropout=0.0,  # No dropout for overfit test
    )

    device = torch.device("cpu")
    model = GPT2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # A Korean sentence to overfit on
    text = "정약용은 조선 후기의 실학자로, 목민심서와 경세유표를 저술했다."
    token_ids = tokenizer.encode(text)
    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

    print(f"  Text: {text}")
    print(f"  Tokens: {len(token_ids)}")
    print(f"  Input shape: {x.shape}\n")

    # Train for a few steps
    model.train()
    initial_loss = None
    for step in range(200):
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f}")

    final_loss = loss.item()
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Reduction:    {initial_loss / final_loss:.1f}x")

    # Verify the model learned something
    model.eval()
    with torch.no_grad():
        prompt = torch.tensor([token_ids[:3]], dtype=torch.long, device=device)
        generated = model.generate(prompt, max_new_tokens=len(token_ids) - 3, temperature=0.5)
        gen_text = tokenizer.decode(generated[0].tolist())
        print(f"\n  Prompt:    {tokenizer.decode(token_ids[:3])}")
        print(f"  Generated: {gen_text}")

    assert final_loss < initial_loss * 0.1, (
        f"Model did not overfit: {final_loss:.4f} >= {initial_loss * 0.1:.4f}"
    )
    print("\n✅ Overfit test passed!")


if __name__ == "__main__":
    test_overfit()
