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

## How to Use

### Important Note for Mac Users (Apple Silicon / MPS)
If your model crashes with an `MPS backend out of memory` error during training, you must limit the memory footprint. Run this in your terminal before training:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```
Then run the training with a smaller batch size and sequence length.

### Quick Test (small model, Wikipedia only, Memory Safe)
```bash
python3 train.py --sources wikipedia --n_layers 4 --n_heads 4 --d_model 256 --d_ff 1024 --epochs 3 --batch_size 2 --seq_len 512
```

### Full Pre-Training (GPT-2 Small config)
If you have a machine with high VRAM (>24GB), you can run the full GPT-2 configuration:
```bash
python3 train.py --epochs 10
```

### Supervised Fine-Tuning (SFT) for Chatbot
After pre-training, turn the model into a chatbot by training it on instruction-following data (`beomi/KoAlpaca-v1.1a`):
```bash
python3 sft.py --base_model checkpoints/best_model.pt --epochs 3
```

### Chat Interface (CLI)
Talk to the fine-tuned model interactively:
```bash
python3 chat.py --model checkpoints/sft/chat_model_best.pt
```

### Web Interface (GUI)
Run a web-based chat application:
```bash
python3 server.py
# Then open http://localhost:8000 in your browser
```

#### Voice Mode (Optional)
The web interface supports WatsonX STT and TTS for voice conversations! 
Before starting the server, export your IBM Cloud credentials:

```bash
export WATSON_STT_API_KEY="your-stt-api-key"
export WATSON_STT_URL="your-stt-url"

export WATSON_TTS_API_KEY="your-tts-api-key"
export WATSON_TTS_URL="your-tts-url"

python3 server.py
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
