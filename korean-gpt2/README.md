# Korean GPT-2

A from-scratch PyTorch implementation of the GPT-2 architecture, trained exclusively on Korean data.

## Data Sources

| Source | Hugging Face ID | Description |
|--------|----------------|-------------|
| Korean Wikipedia | `devngho/korean_wikipedia` | Full Korean Wikipedia articles |
| Korean Web Text | `HAERAE-HUB/KOREAN-WEBTEXT` | Curated Korean internet text |
| Historical Corpus | `seyoungsong/Open-Korean-Historical-Corpus` | 정약용 era Hangeul & Hanja texts |
| YouTube Transcripts | `PleIAs/YouTube-Commons` | Korean-filtered YouTube transcripts |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test (small model, Wikipedia only)
```bash
python3 train.py --sources wikipedia --n_layers 4 --n_heads 4 --d_model 256 --d_ff 1024 --epochs 3
```

### Full Training (GPT-2 Small config)
```bash
python3 train.py --epochs 10
```

### Model Architecture Test
```bash
python3 -m model.gpt2
```

### Data Pipeline Test
```bash
python3 -m data.pipeline
```

## Architecture

Standard GPT-2 architecture:
- Pre-LayerNorm Transformer blocks
- Causal (masked) self-attention
- GELU activation (tanh approximation)
- Learned positional embeddings
- Weight-tied embedding/output layer
- Configurable: layers, heads, embedding dimension
