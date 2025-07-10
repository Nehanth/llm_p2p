#!/usr/bin/env python3
"""
FastAPI route handlers for the distributed inference system
"""

import asyncio
import torch
from fastapi import FastAPI, HTTPException
from typing import Optional

from src.models.data_models import GenerateRequest, GenerateResponse, ShardRequest, ShardResponse
from src.utils.config import ShardConfig
from src.network.p2p_server import P2PServer
from src.utils.helpers import get_logger, dict_to_tensor, tensor_to_dict

logger = get_logger(__name__)

class APIRoutes:
    """FastAPI route handlers for shard server"""
    
    def __init__(self, shard_config: ShardConfig, model, tokenizer, p2p_server: P2PServer):
        self.shard_config = shard_config
        self.model = model
        self.tokenizer = tokenizer
        self.p2p_server = p2p_server
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def setup_routes(self, app: FastAPI):
        """Setup all API routes"""
        
        @app.on_event("startup")
        async def startup_event():
            """Initialize peer discovery on startup"""
            await asyncio.sleep(2)
            await self.p2p_server.discover_peers()
        
        @app.get("/health")
        async def health():
            """Health check endpoint returning shard status"""
            return {
                "status": "healthy",
                "shard_id": self.shard_config.shard_id,
                "layers": f"{self.shard_config.start_layer}-{self.shard_config.end_layer}",
                "is_input": self.shard_config.is_input_shard,
                "is_output": self.shard_config.is_output_shard,
                "device": self.device,
                "model_loaded": self.model is not None
            }
            
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Direct generation endpoint - P2P capability"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Discover peers if not already done
                if not self.p2p_server.peers:
                    await self.p2p_server.discover_peers()
                
                # Generate text using P2P routing
                return await self.p2p_server.generate_text_p2p(request)
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/peers")
        async def get_peers():
            """Get information about discovered peers"""
            await self.p2p_server.discover_peers()
            return self.p2p_server.get_peers_info()

        #process shard that goes back and forth between shards
        @app.post("/process", response_model=ShardResponse)
        async def process_shard(request: ShardRequest):
            """Process shard computation for inter-shard communication"""
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
                            is_final=True,
                            shard_id=self.shard_config.shard_id
                        )
                    else:
                        # Intermediate shard - forward to next shard
                        if self.shard_config.next_shard_url:
                            # Forward to next shard
                            result = await self.p2p_server.forward_to_next_shard(
                                hidden_states, 
                                self.shard_config.next_shard_url
                            )
                            return result
                        else:
                            # No next shard - return tensor data
                            return ShardResponse(
                                tensor_data=tensor_to_dict(hidden_states),
                                is_final=False,
                                shard_id=self.shard_config.shard_id
                            )
                            
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e)) 