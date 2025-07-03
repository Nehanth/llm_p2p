#!/usr/bin/env python3
"""
Sharded model server - hosts a portion of DistilGPT-2 layers
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse
import logging
from typing import Optional, Dict, Any
import numpy as np
import json

from shard_config import DistributedConfig, ShardConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShardedModel(nn.Module):
    """A portion of the DistilGPT-2 model"""
    
    def __init__(self, full_model: GPT2LMHeadModel, start_layer: int, end_layer: int, is_input: bool, is_output: bool):
        super().__init__()
        
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_input_shard = is_input
        self.is_output_shard = is_output
        
        # Extract the layers we need
        if is_input:
            # Input shard includes embeddings + transformer layers
            self.wte = full_model.transformer.wte  # word token embeddings
            self.wpe = full_model.transformer.wpe  # position embeddings
            self.drop = full_model.transformer.drop
            
        # Extract transformer blocks
        self.h = nn.ModuleList()
        for i in range(start_layer, end_layer + 1):
            self.h.append(full_model.transformer.h[i])
            
        if is_output:
            # Output shard includes final layer norm + LM head
            self.ln_f = full_model.transformer.ln_f
            self.lm_head = full_model.lm_head
            
        self.config = full_model.config
        
    def forward(self, inputs):
        """Forward pass through this shard"""
        if self.is_input_shard:
            # Input processing
            if isinstance(inputs, dict):
                input_ids = inputs["input_ids"]
                position_ids = inputs.get("position_ids")
            else:
                input_ids = inputs
                position_ids = None
                
            # Get embeddings
            inputs_embeds = self.wte(input_ids)
            
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            position_embeds = self.wpe(position_ids)
            hidden_states = self.drop(inputs_embeds + position_embeds)
        else:
            # Intermediate shard - input is hidden states
            hidden_states = inputs
            
        # Pass through transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
                
        if self.is_output_shard:
            # Final processing
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        else:
            return hidden_states

class TensorData(BaseModel):
    """Pydantic model for tensor data transfer"""
    data: list
    shape: list
    dtype: str

class ShardRequest(BaseModel):
    """Request format for shard processing"""
    tensor_data: Optional[TensorData] = None
    input_ids: Optional[list] = None
    is_initial: bool = False

class ShardResponse(BaseModel):
    """Response format from shard processing"""
    tensor_data: Optional[TensorData] = None
    logits: Optional[list] = None
    is_final: bool = False

def tensor_to_dict(tensor: torch.Tensor) -> TensorData:
    """Convert tensor to serializable format"""
    return TensorData(
        data=tensor.detach().cpu().numpy().tolist(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype)
    )

def dict_to_tensor(tensor_data: TensorData, device: str) -> torch.Tensor:
    """Convert serializable format back to tensor"""
    data = np.array(tensor_data.data).reshape(tensor_data.shape)
    tensor = torch.from_numpy(data)
    
    # Convert dtype
    if "float" in tensor_data.dtype:
        tensor = tensor.float()
    elif "long" in tensor_data.dtype:
        tensor = tensor.long()
    
    return tensor.to(device)

class ShardServer:
    def __init__(self, shard_config: ShardConfig, model_config: DistributedConfig):
        self.shard_config = shard_config
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # FastAPI app
        self.app = FastAPI(
            title=f"Shard {shard_config.shard_id} Server",
            description=f"Distributed DistilGPT-2 Shard (Layers {shard_config.start_layer}-{shard_config.end_layer})"
        )
        
        self.setup_routes()
        
    def load_model(self):
        """Load the full model and extract our shard"""
        logger.info(f"Loading shard {self.shard_config.shard_id} (layers {self.shard_config.start_layer}-{self.shard_config.end_layer})")
        
        # Load full model
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            torch_dtype=torch.float32
        )
        
        # Load tokenizer (needed for input shard)
        if self.shard_config.is_input_shard:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                cache_dir=self.model_config.cache_dir
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create sharded model
        self.model = ShardedModel(
            full_model,
            self.shard_config.start_layer,
            self.shard_config.end_layer,
            self.shard_config.is_input_shard,
            self.shard_config.is_output_shard
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Shard {self.shard_config.shard_id} loaded successfully on {self.device}")
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "shard_id": self.shard_config.shard_id,
                "layers": f"{self.shard_config.start_layer}-{self.shard_config.end_layer}",
                "is_input": self.shard_config.is_input_shard,
                "is_output": self.shard_config.is_output_shard,
                "device": self.device,
                "model_loaded": self.model is not None
            }
            
        @self.app.post("/process", response_model=ShardResponse)
        async def process_shard(request: ShardRequest):
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
                
            try:
                with torch.no_grad():
                    if self.shard_config.is_input_shard and request.is_initial:
                        # Initial processing - convert input_ids to hidden states
                        input_ids = torch.tensor([request.input_ids], dtype=torch.long).to(self.device)
                        hidden_states = self.model({"input_ids": input_ids})
                    else:
                        # Intermediate processing - process hidden states
                        hidden_states = dict_to_tensor(request.tensor_data, self.device)
                        hidden_states = self.model(hidden_states)
                    
                    if self.shard_config.is_output_shard:
                        # Final shard - return logits
                        return ShardResponse(
                            logits=hidden_states.detach().cpu().numpy().tolist(),
                            is_final=True
                        )
                    else:
                        # Intermediate shard - forward to next shard
                        if self.shard_config.next_shard_url:
                            # Forward to next shard
                            tensor_data = tensor_to_dict(hidden_states)
                            next_request = ShardRequest(tensor_data=tensor_data)
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    f"{self.shard_config.next_shard_url}/process",
                                    json=next_request.dict()
                                ) as response:
                                    result = await response.json()
                                    return ShardResponse(**result)
                        else:
                            # No next shard - return tensor data
                            return ShardResponse(
                                tensor_data=tensor_to_dict(hidden_states),
                                is_final=False
                            )
                            
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description="Run a model shard server")
    parser.add_argument("--shard-id", type=int, required=True, help="Shard ID to run")
    parser.add_argument("--config", type=str, default="shard_config.json", help="Config file path")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and args.config != "shard_config.json":
        config = DistributedConfig.load_config(args.config)
    else:
        from shard_config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    shard_config = config.get_shard_config(args.shard_id)
    
    # Create and setup server
    server = ShardServer(shard_config, config)
    server.load_model()
    
    print(f"🚀 Starting Shard {args.shard_id} Server...")
    print(f"📊 Layers: {shard_config.start_layer}-{shard_config.end_layer}")
    print(f"🌐 Server: http://{shard_config.host}:{shard_config.port}")
    print(f"🔧 Device: {server.device}")
    
    # Run server
    uvicorn.run(
        server.app,
        host=shard_config.host,
        port=shard_config.port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 