#!/usr/bin/env python3
"""
FastAPI server for DistilGPT-2 model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import uvicorn
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DistilGPT-2 API",
    description="A FastAPI server for text generation using DistilGPT-2",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input text prompt for generation")
    max_length: int = Field(default=50, description="Maximum length of generated text", ge=1, le=512)
    temperature: float = Field(default=0.7, description="Temperature for sampling", ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling parameter", ge=0.1, le=1.0)
    top_k: int = Field(default=50, description="Top-k sampling parameter", ge=1, le=200)
    num_return_sequences: int = Field(default=1, description="Number of sequences to generate", ge=1, le=5)
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty", ge=1.0, le=2.0)

class TextGenerationResponse(BaseModel):
    generated_texts: List[str]
    prompt: str
    parameters: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

def load_model():
    """Load the DistilGPT-2 model and tokenizer"""
    global model, tokenizer
    
    try:
        model_name = "distilgpt2"
        cache_dir = "./models"
        
        logger.info(f"Loading model from {cache_dir}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Set pad token to eos token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model parameters: {model.num_parameters():,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device
    )

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using DistilGPT-2"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                num_return_sequences=request.num_return_sequences,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode generated texts
        generated_texts = []
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return TextGenerationResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            parameters=request.dict()
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DistilGPT-2 FastAPI Server",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "model": "distilgpt2",
        "device": device
    }

if __name__ == "__main__":
    print("🚀 Starting DistilGPT-2 FastAPI Server...")
    print(f"🔧 Device: {device}")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "serve_model:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 