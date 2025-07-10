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
            # SHARD 0 (INPUT SHARD)
            # Flow: User query -> Shard 0 -> Token IDs -> Embeddings -> Hidden States
            
            # Input processing
            if isinstance(inputs, dict):
                input_ids = inputs["input_ids"]  # e.g., [15496, 350, 17, 47] for "Hello P2P"
                position_ids = inputs.get("position_ids")
            else:
                input_ids = inputs
                position_ids = None
                
            # STEP 1: Convert token IDs to embeddings (numbers -> vectors)
            # Token IDs [15496, 350, 17, 47] -> Embedding vectors [768-dim each]
            inputs_embeds = self.wte(input_ids)
            
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            # STEP 2: Add position information (where each token is in the sequence)
            position_embeds = self.wpe(position_ids)
            
            # STEP 3: Combine word + position embeddings = HIDDEN STATES (thought vectors)
            hidden_states = self.drop(inputs_embeds + position_embeds)
        else:
            # SHARD 1 (OUTPUT SHARD)
            # Flow: Receive hidden states from Shard 0 -> Process -> Generate logits
            
            # Intermediate shard - input is hidden states from previous shard
            hidden_states = inputs  # Already processed hidden states from Shard 0
            
        # BOTH SHARDS: Process through transformer blocks
        # Flow: Hidden states -> Transformer layers -> Updated hidden states
        # Each transformer block does attention + feed-forward processing
        for block in self.h:
            outputs = block(hidden_states)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]  # Extract hidden states from tuple
            else:
                hidden_states = outputs
                
        if self.is_output_shard:
            # SHARD 1 (OUTPUT SHARD) FINAL STEP
            # Flow: Hidden states -> Layer norm -> LM head -> Logits (50,257 probabilities) -> vocab translation
            
            # Final processing
            hidden_states = self.ln_f(hidden_states)  # Normalize the hidden states
            logits = self.lm_head(hidden_states)      # Convert to vocabulary probabilities
            return logits  # Send back to Shard 0 for token sampling/translation
        else:
            # SHARD 0 (INPUT SHARD) FINAL STEP
            # Flow: Return hidden states -> Send to Shard 1 via P2P
            return hidden_states  # Send to next shard for further processing

class ModelLoader:
    """Handles model loading and initialization for distributed P2P system"""
    
    def __init__(self, shard_config: ShardConfig, model_config: Config):
        # Setup: Each shard gets its own ModelLoader with specific configuration
        self.shard_config = shard_config    
        self.model_config = model_config  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        
    def load_model(self):
        """Load the full model and extract our shard"""
        # Flow: Download complete model -> Extract only our shard's layers
        logger.info(f"Loading shard {self.shard_config.shard_id} (layers {self.shard_config.start_layer}-{self.shard_config.end_layer})")
        
        # This gets the complete GPT2LMHeadModel with all 6 layers + embeddings + LM head
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,  # "distilgpt2"
            cache_dir=self.model_config.cache_dir,  # "./models" 
            torch_dtype=torch.float32  # Use float32 for precision
        )
        
        # STEP 2: CREATE SHARDED MODEL
        # Flow: Full model -> Extract specific layers -> Create shard-specific model
        # This is where the magic happens - we split the model!
        model = ShardedModel(
            full_model,  
            self.shard_config.start_layer,  # Shard 0: 0, Shard 1: 3
            self.shard_config.end_layer,    # Shard 0: 2, Shard 1: 5
            self.shard_config.is_input_shard, 
            self.shard_config.is_output_shard  
        )
        
        # STEP 3: PREPARE FOR INFERENCE 

        model.to(self.device)  #cuda or cpu
        model.eval() #eval means inference
        
        logger.info(f"Shard {self.shard_config.shard_id} loaded successfully on {self.device}")
        return model  # Return the prepared shard
    
    def load_tokenizer(self):
        """Load tokenizer (only for input shard)"""
        # TOKENIZER LOADING for shard 0
        # Flow: Only Shard 0 needs tokenizer (text -> tokens, tokens -> text) (full translation)
        
        if self.shard_config.is_input_shard:
            # convet text to token id and back
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,  # "distilgpt2"
                cache_dir=self.model_config.cache_dir  # "./models"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer 
        
        return None 