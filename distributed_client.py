#!/usr/bin/env python3
"""
Distributed client for coordinating text generation across sharded models
"""

import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import List, Optional
from transformers import AutoTokenizer
import torch
import numpy as np

from shard_config import DistributedConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGenerationRequest(BaseModel):
    """Request format for text generation"""
    prompt: str = Field(..., description="The input text prompt for generation")
    max_length: int = Field(default=50, description="Maximum length of generated text", ge=1, le=512)
    temperature: float = Field(default=0.7, description="Temperature for sampling", ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling parameter", ge=0.1, le=1.0)
    top_k: int = Field(default=50, description="Top-k sampling parameter", ge=1, le=200)
    num_return_sequences: int = Field(default=1, description="Number of sequences to generate", ge=1, le=5)
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty", ge=1.0, le=2.0)

class TextGenerationResponse(BaseModel):
    """Response format for text generation"""
    generated_texts: List[str]
    prompt: str
    parameters: dict
    processing_time: float
    shards_used: List[int]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    shards_status: dict
    total_shards: int

class DistributedClient:
    """Client for coordinating distributed inference"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.tokenizer = None
        
        # FastAPI app
        self.app = FastAPI(
            title="Distributed DistilGPT-2 Client",
            description="Coordinator for distributed model inference"
        )
        
        self.setup_routes()
        
    def load_tokenizer(self):
        """Load tokenizer for text processing"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("✅ Tokenizer loaded")
        
    async def check_shards_health(self) -> dict:
        """Check health of all shards"""
        health_status = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for shard in self.config.shards:
                url = f"http://{shard.host}:{shard.port}/health"
                tasks.append(self._check_shard_health(session, shard.shard_id, url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                shard_id = self.config.shards[i].shard_id
                if isinstance(result, Exception):
                    health_status[shard_id] = {"status": "unhealthy", "error": str(result)}
                else:
                    health_status[shard_id] = result
                    
        return health_status
    
    async def _check_shard_health(self, session: aiohttp.ClientSession, shard_id: int, url: str):
        """Check health of a single shard"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unhealthy", "http_status": response.status}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        """Generate text using distributed shards with autoregressive generation"""
        import time
        start_time = time.time()
        
        if self.tokenizer is None:
            raise HTTPException(status_code=503, detail="Tokenizer not loaded")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(request.prompt, add_special_tokens=True)
        
        # Find input shard
        input_shard = None
        for shard in self.config.shards:
            if shard.is_input_shard:
                input_shard = shard
                break
                
        if input_shard is None:
            raise HTTPException(status_code=500, detail="No input shard found")
        
        generated_texts = []
        
        # Generate multiple sequences
        for seq_idx in range(request.num_return_sequences):
            current_ids = input_ids.copy()
            
            # Autoregressive generation loop
            for step in range(request.max_length - len(input_ids)):
                # Process current sequence through shards
                async with aiohttp.ClientSession() as session:
                    url = f"http://{input_shard.host}:{input_shard.port}/process"
                    
                    shard_request = {
                        "input_ids": current_ids,
                        "is_initial": True
                    }
                    
                    async with session.post(url, json=shard_request) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise HTTPException(status_code=500, detail=f"Shard processing failed: {error_text}")
                        
                        result = await response.json()
                
                # Extract logits
                if not result.get("is_final", False):
                    raise HTTPException(status_code=500, detail="Did not receive final output from shards")
                
                logits = torch.tensor(result["logits"])
                
                # Sample next token
                if request.do_sample:
                    # Apply temperature
                    logits_processed = logits[0, -1] / request.temperature
                    
                    # Apply top-k filtering
                    if request.top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits_processed, request.top_k)
                        logits_filtered = torch.full_like(logits_processed, float('-inf'))
                        logits_filtered[top_k_indices] = top_k_logits
                    else:
                        logits_filtered = logits_processed
                    
                    # Apply top-p filtering
                    if request.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits_filtered, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > request.top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits_filtered[indices_to_remove] = float('-inf')
                    
                    # Apply repetition penalty
                    if request.repetition_penalty != 1.0:
                        for token_id in set(current_ids):
                            if logits_filtered[token_id] < 0:
                                logits_filtered[token_id] *= request.repetition_penalty
                            else:
                                logits_filtered[token_id] /= request.repetition_penalty
                    
                    # Sample from filtered distribution
                    probs = torch.softmax(logits_filtered, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits[0, -1], dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                
                # Add token to sequence
                current_ids.append(next_token_id)
                
                # Check for EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
            
            # Decode generated sequence
            generated_text = self.tokenizer.decode(current_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        processing_time = time.time() - start_time
        shards_used = [shard.shard_id for shard in self.config.shards]
        
        return TextGenerationResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            parameters=request.dict(),
            processing_time=processing_time,
            shards_used=shards_used
        )
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Check health of all shards"""
            shards_status = await self.check_shards_health()
            
            all_healthy = all(
                status.get("status") == "healthy" 
                for status in shards_status.values()
            )
            
            return HealthResponse(
                status="healthy" if all_healthy else "degraded",
                shards_status=shards_status,
                total_shards=len(self.config.shards)
            )
        
        @self.app.post("/generate", response_model=TextGenerationResponse)
        async def generate_text_endpoint(request: TextGenerationRequest):
            """Generate text using distributed shards"""
            return await self.generate_text(request)
        
        @self.app.get("/")
        async def root():
            """Root endpoint with information"""
            return {
                "message": "Distributed DistilGPT-2 Client",
                "shards": len(self.config.shards),
                "endpoints": {
                    "health": "/health",
                    "generate": "/generate",
                    "docs": "/docs"
                }
            }
        
        @self.app.get("/shards")
        async def list_shards():
            """List all configured shards"""
            return {
                "shards": [
                    {
                        "shard_id": shard.shard_id,
                        "layers": f"{shard.start_layer}-{shard.end_layer}",
                        "url": f"http://{shard.host}:{shard.port}",
                        "is_input": shard.is_input_shard,
                        "is_output": shard.is_output_shard
                    }
                    for shard in self.config.shards
                ]
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run distributed client")
    parser.add_argument("--config", type=str, default="shard_config.json", help="Config file path")
    parser.add_argument("--port", type=int, default=9000, help="Client port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Client host")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = DistributedConfig.load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using default config")
        from shard_config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    # Create client
    client = DistributedClient(config)
    client.load_tokenizer()
    
    print("🌐 Starting Distributed DistilGPT-2 Client...")
    print(f"📊 Managing {len(config.shards)} shards")
    print(f"🔧 Client URL: http://{args.host}:{args.port}")
    print(f"📖 Documentation: http://{args.host}:{args.port}/docs")
    
    # Run client
    uvicorn.run(
        client.app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 