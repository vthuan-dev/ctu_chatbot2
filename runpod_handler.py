"""
RunPod Serverless Handler for CTU Chatbot
"""

import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import time
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
model = None
tokenizer = None

def load_model():
    """Load model and tokenizer"""
    global model, tokenizer
    
    if model is not None:
        return
    
    logger.info("Loading model...")
    
    # First try using Unsloth FastLanguageModel (recommended for Unsloth-trained models)
    model_name = os.getenv("MODEL_NAME", "thuanhero1/llama3-8b-finetuned-ctu")
    
    try:
        logger.info(f"Attempting to load with Unsloth FastLanguageModel: {model_name}")
        
        # Use Unsloth's FastLanguageModel to load the fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        # Enable fast inference mode
        FastLanguageModel.for_inference(model)
        
        logger.info("Model loaded successfully with Unsloth!")
        
    except Exception as e:
        logger.warning(f"Failed to load with Unsloth: {e}")
        logger.info("Trying standard transformers approach...")
        
        try:
            logger.info(f"Attempting to load fine-tuned model with transformers: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load fine-tuned model directly
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Fine-tuned model loaded successfully with transformers!")
            
        except Exception as e2:
            logger.warning(f"Failed to load fine-tuned model with transformers: {e2}")
            logger.info("Falling back to base model + LoRA approach...")
            
            # Fallback: Load base model + LoRA
            base_model_name = os.getenv("BASE_MODEL_NAME", "unsloth/llama-3-8b-instruct-bnb-4bit")
            lora_adapter_name = os.getenv("LORA_ADAPTER_NAME", "thuanhero1/llama3-8b-finetuned-ctu")
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # Try Unsloth first for base model
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    max_seq_length=2048,
                    dtype=torch.float16,
                    load_in_4bit=True,
                )
                
                # Load LoRA adapter
                logger.info(f"Loading LoRA adapter: {lora_adapter_name}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, lora_adapter_name)
                
                # Enable fast inference
                FastLanguageModel.for_inference(model)
                
                logger.info("Base model + LoRA loaded successfully with Unsloth!")
                
            except Exception as e3:
                logger.warning(f"Unsloth base model failed: {e3}")
                logger.info("Using standard transformers for base model...")
                
                # Load tokenizer from base model
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load base model with optimization
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                # Load LoRA adapter
                logger.info(f"Loading LoRA adapter: {lora_adapter_name}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, lora_adapter_name)
                
                logger.info("Base model + LoRA loaded successfully with transformers!")
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    logger.info("Model loading completed!")

def generate_response(prompt, max_tokens=256, temperature=0.7, top_p=0.9):
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
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant<|end_header_id|>" in full_response:
        response = full_response.split("assistant<|end_header_id|>")[-1].strip()
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0].strip()
    else:
        response = full_response
    
    # Count tokens
    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    
    return {
        "response": response,
        "tokens_generated": tokens_generated,
        "response_time": round(response_time, 2)
    }

def handler(job):
    """
    RunPod serverless handler function
    """
    job_input = job["input"]
    
    # Load model if not already loaded
    load_model()
    
    # Extract parameters
    prompt = job_input.get("prompt", "")
    max_tokens = job_input.get("max_tokens", 256)
    temperature = job_input.get("temperature", 0.7)
    top_p = job_input.get("top_p", 0.9)
    
    if not prompt:
        return {"error": "No prompt provided"}
    
    try:
        # Generate response
        result = generate_response(prompt, max_tokens, temperature, top_p)
        
        return {
            "output": result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

# RunPod endpoint
runpod.serverless.start({"handler": handler})