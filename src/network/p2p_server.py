#!/usr/bin/env python3
"""
P2P server logic for distributed inference - handles networking and text generation
"""

import asyncio
import aiohttp
import time
import torch
from typing import Optional, Dict, List
from fastapi import HTTPException

from src.models.data_models import PeerInfo, GenerateRequest, GenerateResponse, ShardRequest, ShardResponse
from src.utils.config import ShardConfig
from src.utils.helpers import get_logger, tensor_to_dict, dict_to_tensor

logger = get_logger(__name__)

class P2PServer:
    """Handles peer-to-peer networking, communication, and text generation"""
    
    def __init__(self, shard_config: ShardConfig, model, tokenizer):
        self.shard_config = shard_config
        self.model = model
        self.tokenizer = tokenizer
        self.peers = {}  # Known peers in the network
        self.network_topology = {}  # Network topology for routing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def discover_peers(self):
        """Discover other peers in the network using hardcoded peer list"""
        logger.info("Discovering peers...")
        
        # For now, use hardcoded peer discovery
        # In a real P2P system, this would use DHT/gossip protocol
        potential_peers = [
            {"host": "172.31.42.169", "port": 8000},
            {"host": "172.31.34.102", "port": 8000},
        ]
        
        for peer_info in potential_peers:
            # Skip self
            if (peer_info["host"] == self.shard_config.host and 
                peer_info["port"] == self.shard_config.port):
                continue
                
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{peer_info['host']}:{peer_info['port']}/health",
                        timeout=5
                    ) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            peer = PeerInfo(
                                shard_id=health_data["shard_id"],
                                layers=health_data["layers"],
                                host=peer_info["host"],
                                port=peer_info["port"],
                                is_input=health_data["is_input"],
                                is_output=health_data["is_output"],
                                status="healthy"
                            )
                            self.peers[peer.shard_id] = peer
                            logger.info(f"Discovered peer: Shard {peer.shard_id} at {peer.host}:{peer.port}")
            except Exception as e:
                logger.warning(f"Failed to connect to peer {peer_info['host']}:{peer_info['port']}: {e}")
                
        logger.info(f"Discovered {len(self.peers)} peers")
        
    async def find_input_shard(self) -> Optional[PeerInfo]:
        """Find the input shard in the network"""
        for peer in self.peers.values():
            if peer.is_input:
                return peer
        return None
        
    async def find_output_shard(self) -> Optional[PeerInfo]:
        """Find the output shard in the network"""
        for peer in self.peers.values():
            if peer.is_output:
                return peer
        return None
        
    async def route_to_input_shard(self, request: GenerateRequest) -> GenerateResponse:
        """Route generation request to input shard"""
        input_shard = await self.find_input_shard()
        if not input_shard:
            raise HTTPException(status_code=503, detail="No input shard available")
            
        logger.info(f"Routing to input shard: {input_shard.host}:{input_shard.port}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{input_shard.host}:{input_shard.port}/generate",
                json=request.dict()
            ) as response:
                result = await response.json()
                return GenerateResponse(**result)
                
    async def forward_to_next_shard(self, hidden_states, next_shard_url: str) -> ShardResponse:
        """Forward hidden states to the next shard"""
        tensor_data = tensor_to_dict(hidden_states)
        next_request = ShardRequest(tensor_data=tensor_data)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{next_shard_url}/process",
                json=next_request.dict()
            ) as response:
                result = await response.json()
                return ShardResponse(**result)
                
    def get_peers_info(self) -> Dict:
        """Get information about discovered peers"""
        return {
            "peers": [peer.dict() for peer in self.peers.values()],
            "total_peers": len(self.peers),
            "self_shard_id": self.shard_config.shard_id
        }
        
    async def generate_text_p2p(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text using P2P routing between shards"""
        start_time = time.time()
        shards_used = []
        
        # Step 1: Find input shard
        input_shard = await self.find_input_shard()
        if not input_shard and not self.shard_config.is_input_shard:
            raise HTTPException(status_code=503, detail="No input shard available")
            
        # Step 2: Route to input shard or process locally
        if self.shard_config.is_input_shard:
            return await self._generate_as_input_shard(request, start_time, shards_used)
        else:
            return await self.route_to_input_shard(request)
            
    async def _generate_as_input_shard(self, request: GenerateRequest, start_time: float, shards_used: List[int]) -> GenerateResponse:
        """Generate text when we are the input shard"""
        logger.info("Processing as input shard")
        shards_used.append(self.shard_config.shard_id)
        
        # Tokenize input
        inputs = self.tokenizer.encode(request.prompt, return_tensors="pt").to(self.device)
        
        # Generate tokens
        generated_tokens = []
        current_input = inputs
        
        for _ in range(request.max_length - len(inputs[0])):
            # Process current input through our shard
            with torch.no_grad():
                hidden_states = self.model({"input_ids": current_input})
                
            # Route to next shard
            if self.shard_config.next_shard_url:
                result = await self.forward_to_next_shard(
                    hidden_states, 
                    self.shard_config.next_shard_url
                )
                
                if result.is_final:
                    # Get logits from final shard
                    logits = torch.tensor(result.logits).to(self.device)
                    
                    # Sample next token (this is translation/sampling step on Shard 0)
                    next_token = self._sample_next_token(logits, request)
                    
                    # Track that Shard 0 did the translation/sampling
                    shards_used.append(self.shard_config.shard_id)
                    
                    # Check for end of sequence
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    generated_tokens.append(next_token.item())
                    current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                
                # Track shards used
                if hasattr(result, 'shard_id') and result.shard_id:
                    shards_used.append(result.shard_id)
                    
        # Decode generated text
        if generated_tokens:
            full_tokens = torch.cat([inputs, torch.tensor([generated_tokens]).to(self.device)], dim=1)
        else:
            full_tokens = inputs
            
        generated_text = self.tokenizer.decode(full_tokens[0], skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        
        return GenerateResponse(
            generated_texts=[generated_text],
            prompt=request.prompt,
            processing_time=processing_time,
            shards_used=shards_used
        )
    
    def _sample_next_token(self, logits: torch.Tensor, request: GenerateRequest) -> torch.Tensor:
        """Sample next token using various sampling strategies"""
        # Handle logits dimensions properly
        if logits.dim() == 3:
            logits = logits[0, -1, :]  # [batch, seq_len, vocab] -> [vocab]
        elif logits.dim() == 2:
            logits = logits[-1, :]     # [seq_len, vocab] -> [vocab]
        
        if request.do_sample:
            # Apply temperature
            logits = logits / request.temperature
            
            # Apply top_k filtering
            if request.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(request.top_k, logits.size(-1)))
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # Apply top_p filtering
            if request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > request.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        return next_token 