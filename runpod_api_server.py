"""
FastAPI server for deploying CTU chatbot on RunPod
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uvicorn
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="CTU Admission Chatbot API",
    description="API for Can Tho University admission consulting",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    response_time: float
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str]

# Global variables
model = None
tokenizer = None
MODEL_NAME = os.getenv("MODEL_NAME", "thuanhero1/llama3-8b-finetuned-ctu")

def load_model():
    """Load model from HuggingFace"""
    global model, tokenizer
    
    logger.info(f"Loading model: {MODEL_NAME}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimization for A5000
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Enable TF32 for A5000
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def generate_response(prompt: str, max_tokens: int, temperature: float, top_p: float) -> tuple:
    """Generate response from model"""
    
    # Format prompt for Llama3
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một trợ lý AI hữu ích, được huấn luyện để trả lời các câu hỏi về Đại học Cần Thơ. Hãy trả lời một cách chính xác, rõ ràng và hữu ích.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    response_time = time.time() - start_time
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response for Llama3
    if "assistant<|end_header_id|>" in full_response:
        response = full_response.split("assistant<|end_header_id|>")[-1].strip()
        # Remove any trailing special tokens
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0].strip()
    else:
        response = full_response
    
    # Count tokens
    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    
    return response, tokens_generated, response_time

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response, tokens, time_taken = generate_response(
            request.message,
            request.max_tokens,
            request.temperature,
            request.top_p
        )
        
        return ChatResponse(
            response=response,
            tokens_generated=tokens,
            response_time=round(time_taken, 2),
            model_name=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=List[ChatResponse])
async def batch_chat(requests: List[ChatRequest]):
    """Batch processing endpoint"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    responses = []
    
    for request in requests:
        try:
            response, tokens, time_taken = generate_response(
                request.message,
                request.max_tokens,
                request.temperature,
                request.top_p
            )
            
            responses.append(ChatResponse(
                response=response,
                tokens_generated=tokens,
                response_time=round(time_taken, 2),
                model_name=MODEL_NAME
            ))
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            responses.append(ChatResponse(
                response=f"Error: {str(e)}",
                tokens_generated=0,
                response_time=0,
                model_name=MODEL_NAME
            ))
    
    return responses

@app.get("/stats")
async def get_stats():
    """Get model and system statistics"""
    
    stats = {
        "model_name": MODEL_NAME,
        "model_loaded": model is not None,
        "device": str(model.device) if model else "N/A",
        "gpu_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        stats.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        })
    
    return stats

if __name__ == "__main__":
    # Run with: python runpod_api_server.py
    # Or with custom port: python runpod_api_server.py --port 8080
    import sys
    
    port = 8000
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])
    
    uvicorn.run(app, host="0.0.0.0", port=port)
