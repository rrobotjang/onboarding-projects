"""
FastAPI Server for Korean GPT-2 Chatbot.

Provides a unified interface:
- Serves static files for the web frontend (HTML/CSS/JS).
- Serves a Server-Sent Events (SSE) endpoint at /api/chat for streaming text generation.
"""

import asyncio
import os
import json
import logging
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ibm_watson import SpeechToTextV1, TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from model.gpt2 import GPT2, GPT2Config
from data.tokenizer import get_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Korean GPT-2 Chat")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.8         # Increased slightly for more variety
    top_k: int = 50
    top_p: float = 0.9              # Nucleus sampling
    repetition_penalty: float = 1.2 # Penalize repeat tokens
    max_new_tokens: int = 256


# --- WatsonX Configuration ---

# Load credentials from environment variables directly or hard-code for testing.
# For production, always use `dotenv` or system environment variables.
WATSON_STT_API_KEY = os.environ.get("WATSON_STT_API_KEY", "")
WATSON_STT_URL = os.environ.get("WATSON_STT_URL", "")

WATSON_TTS_API_KEY = os.environ.get("WATSON_TTS_API_KEY", "")
WATSON_TTS_URL = os.environ.get("WATSON_TTS_URL", "")

# --- Mock Configuration for UI Testing ---
MOCK_MODE = not (WATSON_STT_API_KEY and WATSON_TTS_API_KEY)
if MOCK_MODE:
    logger.info("⚠️ Watson credentials not found. Enabling MOCK MODE for STT/TTS UI testing.")

stt = None
tts = None

# Initialize Watson STT if credentials are provided
if WATSON_STT_API_KEY and WATSON_STT_URL:
    stt_authenticator = IAMAuthenticator(WATSON_STT_API_KEY)
    stt = SpeechToTextV1(authenticator=stt_authenticator)
    stt.set_service_url(WATSON_STT_URL)

# Initialize Watson TTS if credentials are provided
if WATSON_TTS_API_KEY and WATSON_TTS_URL:
    tts_authenticator = IAMAuthenticator(WATSON_TTS_API_KEY)
    tts = TextToSpeechV1(authenticator=tts_authenticator)
    tts.set_service_url(WATSON_TTS_URL)

# --- Model Loading ---

model = None
tokenizer = None
device = None

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@app.on_event("startup")
async def load_model_on_startup():
    await reload_model_logic()

async def reload_model_logic():
    global model, tokenizer, device
    
    # Force CPU for the chatbot to avoid contention with training on MPS
    device = torch.device("cpu")
    logger.info("🖥️ Chatbot utilizing CPU for inference (Training is on MPS)")
    
    tokenizer = get_tokenizer()
    
    # Priority List for Model Loading
    model_paths = [
        "checkpoints/sft/chat_model_best.pt",
        "checkpoints/sft/latest_sft.pt",
        "checkpoints/latest_model.pt",
        "checkpoints/best_model.pt"
    ]
    
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                logger.info(f"📥 Loading model from {path}...")
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                
                config = checkpoint.get("config", GPT2Config(vocab_size=tokenizer.vocab_size))
                model = GPT2(config)
                
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                logger.info(f"✅ Successfully loaded: {path}")
                loaded = True
                break
            except Exception as e:
                logger.error(f"❌ Failed to load {path}: {e}")

    if not loaded:
        logger.warning("⚠️ No checkpoints found. Running with random weights for UI testing.")
        config = GPT2Config(vocab_size=tokenizer.vocab_size, n_layers=2, n_heads=2, d_model=128)
        model = GPT2(config).to(device)
        model.eval()

@app.post("/api/reload")
async def reload_model():
    """Manually reload the model brain without restarting server."""
    try:
        await reload_model_logic()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Routes ---

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "mock_mode": MOCK_MODE,
        "device": str(device) if device else "cpu",
        "model_loaded": model is not None
    }


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> StreamingResponse:
    """
    Accepts conversation history and streams the model's response back via SSE.
    """
    
    # 1. Format the conversation using SFT tokens
    user_token = "<|user|>\n"
    assistant_token = "\n<|assistant|>\n"
    
    prompt = ""
    for msg in request.messages:
        if msg.role == "user":
            prompt += f"{user_token}{msg.content}"
        elif msg.role == "assistant":
            prompt += f"{assistant_token}{msg.content}\n"
            
    # Add prompt trigger for the new assistant response
    prompt += f"{assistant_token}"
    
    logger.info(f"Received chat request prompt (length: {len(prompt)})")

    # 2. Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 3. Create an async generator for Server Sent Events (SSE)
    async def generate_response() -> AsyncGenerator[str, None]:
        nonlocal input_ids
        
        generated_count = 0
        eos_id = tokenizer.eos_token_id
        
        # We manually step the generation loop so we can yield tokens back
        for _ in range(request.max_new_tokens):
            
            # Predict next token
            with torch.no_grad():
                logits, _ = model(input_ids)
                logits = logits[:, -1, :]
                
                # 1. Apply repetition penalty
                # We iterate through previously seen tokens and reduce their probability in the logits
                if request.repetition_penalty != 1.0:
                    # Get unique tokens in the current sequence
                    unique_tokens = set(input_ids[0].tolist())
                    for token_id in unique_tokens:
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= request.repetition_penalty
                        else:
                            logits[0, token_id] *= request.repetition_penalty

                # 2. Scale by temperature
                logits = logits / request.temperature
                
                # 3. Apply top-k
                if request.top_k is not None and request.top_k > 0:
                    v, _ = torch.topk(logits, min(request.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                
                # 4. Apply top-p (nucleus sampling)
                if request.top_p is not None and request.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > request.top_p
                    # Shift the shift to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float("-inf")

                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            token_id = idx_next.item()
            
            # Stop condition
            if token_id == eos_id:
                break
                
            # Decode just this token 
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
            generated_count += 1
            
            # Extract just the newly generated text to yield
            new_tokens_idx = input_ids.size(1) - generated_count
            response_ids = input_ids[0, new_tokens_idx:]
            current_text = tokenizer.decode(response_ids.tolist())

            # --- Reasoning Simulation Trigger ---
            # If the user asks about thinking or reasoning, we inject a mock CoT response for UI testing
            if "thinking" in prompt.lower() or "생각" in prompt:
                mock_cot = "<|thought|>\n사용자의 질문을 분석 중입니다. 한국어 GPT-2 모델의 추론 능력을 테스트하기 위해 단계별 설명을 준비하고 있습니다.\n1. 질문의 의도 파악\n2. 관련 지식 검색\n3. 자연스러운 문장 생성\n<|assistant|>\n정상적으로 추론 과정을 출력하고 있습니다. 이 블록은 'Thinking Process'로 표시됩니다."
                current_text = mock_cot
                payload = json.dumps({"text": current_text, "done": True}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                return

            # We yield the full text update as a JSON string
            payload = json.dumps({"text": current_text, "done": False}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            
            # Give back event loop control slightly so HTTP chunks can flush
            await asyncio.sleep(0.01)

        # Final message indicating completion
        payload = json.dumps({"text": current_text, "done": True}, ensure_ascii=False)
        yield f"data: {payload}\n\n"

    # StreamingResponse with text/event-stream content type is used for SSE
    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Accepts an audio file via multipart/form-data and uses WatsonX STT to transcribe it.
    Returns the transcribed text.
    """
    if not stt:
        if MOCK_MODE:
            logger.info("STT Mock: Returning placeholder text.")
            return {"text": "이것은 음성 인식 테스트를 위한 모킹된 텍스트입니다."}
        return {"error": "WatsonX STT credentials not configured."}
        
    try:
        content = await audio.read()
        
        # Call Watson STT with Korean narrow-band or broadband model based on your audio
        response = stt.recognize(
            audio=content,
            content_type=audio.content_type or 'audio/webm',
            model='ko-KR_BroadbandModel', # Standard Korean model
            timestamps=False,
            word_confidence=False
        ).get_result()
        
        # Parse the response to get the highest confidence transcript
        if response and response.get('results') and len(response['results']) > 0:
            transcript = ""
            for result in response['results']:
                transcript += result['alternatives'][0]['transcript']
            return {"text": transcript.strip()}
            
        return {"text": ""}
        
    except Exception as e:
        logger.error(f"STT Error: {str(e)}")
        return {"error": str(e)}


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """
    Accepts text and uses WatsonX TTS to generate an audio file.
    Returns the audio byte stream.
    """
    if not tts:
        if MOCK_MODE:
            logger.info("TTS Mock: Returning silent placeholder audio.")
            # Tiny silent MP3 for testing UI flow
            silent_mp3 = b'\xff\xfb\x90D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            return Response(content=silent_mp3, media_type="audio/mp3")
        return {"error": "WatsonX TTS credentials not configured."}
        
    try:
        response = tts.synthesize(
            request.text,
            voice='ko-KR_YunaVoice', # Korean female voice (or ko-KR_YoungmiVoice)
            accept='audio/mp3'
        ).get_result()
        
        audio_content = response.content
        return Response(content=audio_content, media_type="audio/mp3")
        
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
