#!/usr/bin/env python3
"""
Configuration for distributed model sharding
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ShardConfig:
    """Configuration for a single shard"""
    shard_id: int
    start_layer: int
    end_layer: int
    host: str
    port: int
    is_input_shard: bool = False
    is_output_shard: bool = False
    next_shard_url: Optional[str] = None
    public_ip: Optional[str] = None  # Added for multi-node support

class DistributedConfig:
    """Configuration for the entire distributed system"""
    
    def __init__(self):
        """Initialize default distributed configuration"""
        # DistilGPT-2 has 6 transformer layers (0-5)
        self.total_layers = 6
        self.model_name = "distilgpt2"
        self.cache_dir = "./models"
        
        # Default 2-shard configuration
        self.shards = [
            ShardConfig(
                shard_id=0,
                start_layer=0,
                end_layer=2,  # layers 0, 1, 2
                host="0.0.0.0",
                port=8000,
                is_input_shard=True,
                next_shard_url="http://localhost:8001"
            ),
            ShardConfig(
                shard_id=1,
                start_layer=3,
                end_layer=5,  # layers 3, 4, 5
                host="0.0.0.0",
                port=8001,
                is_output_shard=True
            )
        ]
    
    def get_shard_config(self, shard_id: int) -> ShardConfig:
        """Get configuration for a specific shard"""
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise ValueError(f"Shard {shard_id} not found")
    
    def save_config(self, filename: str):
        """Save configuration to JSON file"""
        config_dict = {
            "total_layers": self.total_layers,
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "start_layer": shard.start_layer,
                    "end_layer": shard.end_layer,
                    "host": shard.host,
                    "port": shard.port,
                    "is_input_shard": shard.is_input_shard,
                    "is_output_shard": shard.is_output_shard,
                    "next_shard_url": shard.next_shard_url
                }
                for shard in self.shards
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filename: str):
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.total_layers = config_dict["total_layers"]
        config.model_name = config_dict["model_name"]
        config.cache_dir = config_dict["cache_dir"]
        
        config.shards = []
        for shard_dict in config_dict["shards"]:
            shard = ShardConfig(**shard_dict)
            config.shards.append(shard)
        
        return config

# Default configuration
DEFAULT_CONFIG = DistributedConfig()

def main():
    """Main function to generate and display default configuration"""
    # Save default config
    DEFAULT_CONFIG.save_config("shard_config.json")
    print("Saved default shard configuration to shard_config.json")
    
    # Print shard info
    print("\nShard Configuration:")
    for shard in DEFAULT_CONFIG.shards:
        print(f"  Shard {shard.shard_id}: Layers {shard.start_layer}-{shard.end_layer} on {shard.host}:{shard.port}")
        if shard.is_input_shard:
            print(f"    → Input shard, forwards to: {shard.next_shard_url}")
        if shard.is_output_shard:
            print(f"    → Output shard")

if __name__ == "__main__":
    main() 