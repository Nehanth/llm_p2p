#!/usr/bin/env python3
"""
Simple configuration loader for YAML-based config
"""

import yaml
from dataclasses import dataclass
from typing import Optional, List

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
    public_ip: Optional[str] = None
    next_shard_url: Optional[str] = None

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            self.data = yaml.safe_load(f)
        
        # Model config
        self.model_name = self.data['model']['name']
        self.total_layers = self.data['model']['total_layers']
        self.cache_dir = self.data['model']['cache_dir']
        
        # Network config
        self.instance1_ip = self.data['network']['instance1_ip']
        self.instance2_ip = self.data['network']['instance2_ip']
        
        # Build shards
        self.shards = []
        for shard_data in self.data['shards']:
            shard = ShardConfig(**shard_data)
            
            # Set public IP based on shard ID
            if shard.shard_id == 0:
                shard.public_ip = self.instance1_ip
                shard.next_shard_url = f"http://{self.instance2_ip}:{shard.port}"
            else:
                shard.public_ip = self.instance2_ip
                
            self.shards.append(shard)
    
    def get_shard_config(self, shard_id: int) -> ShardConfig:
        """Get configuration for a specific shard"""
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise ValueError(f"Shard {shard_id} not found")
    


def main():
    """Test the config loader"""
    config = Config()
    
    print("Configuration loaded:")
    print(f"Model: {config.model_name}")
    print(f"Instances: {config.instance1_ip}, {config.instance2_ip}")
    print(f"Shards: {len(config.shards)}")
    
    for shard in config.shards:
        print(f"  Shard {shard.shard_id}: Layers {shard.start_layer}-{shard.end_layer}")

if __name__ == "__main__":
    main() 