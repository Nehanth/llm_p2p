#!/usr/bin/env python3
"""
Data models for the distributed DistilGPT-2 system
"""

from pydantic import BaseModel
from typing import Optional, List

class GenerateRequest(BaseModel):
    """Request format for direct generation"""
    prompt: str
    max_length: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    num_return_sequences: int = 1

class GenerateResponse(BaseModel):
    """Response format for direct generation"""
    generated_texts: List[str]
    prompt: str
    processing_time: float
    shards_used: List[int]

class PeerInfo(BaseModel):
    """Information about a peer shard"""
    shard_id: int
    layers: str
    host: str
    port: int
    is_input: bool
    is_output: bool
    status: str = "unknown"

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