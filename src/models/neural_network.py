#!/usr/bin/env python3
"""
Sharded neural network model for distributed inference
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from src.utils.config import Config, ShardConfig
from src.utils.helpers import get_logger

logger = get_logger(__name__)

class ShardedModel(nn.Module):
    """A portion of the DistilGPT-2 model"""
    
    def __init__(self, full_model: GPT2LMHeadModel, start_layer: int, end_layer: int, is_input: bool, is_output: bool):
        """Initialize sharded model with specific layers"""
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

class ModelLoader:
    """Handles model loading and initialization"""
    
    def __init__(self, shard_config: ShardConfig, model_config: Config):
        self.shard_config = shard_config
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the full model and extract our shard"""
        logger.info(f"Loading shard {self.shard_config.shard_id} (layers {self.shard_config.start_layer}-{self.shard_config.end_layer})")
        
        # Load full model
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            torch_dtype=torch.float32
        )
        
        # Create sharded model
        model = ShardedModel(
            full_model,
            self.shard_config.start_layer,
            self.shard_config.end_layer,
            self.shard_config.is_input_shard,
            self.shard_config.is_output_shard
        )
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Shard {self.shard_config.shard_id} loaded successfully on {self.device}")
        return model
    
    def load_tokenizer(self):
        """Load tokenizer (only for input shard)"""
        if self.shard_config.is_input_shard:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                cache_dir=self.model_config.cache_dir
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        return None 