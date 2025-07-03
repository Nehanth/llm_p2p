#!/usr/bin/env python3
"""
Multi-node configuration for distributed model sharding across different instances
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from shard_config import ShardConfig

class MultiNodeConfig:
    """Configuration for multi-node distributed setup"""
    
    def __init__(self):
        """Initialize multi-node configuration with default values"""
        # DistilGPT-2 has 6 transformer layers (0-5)
        self.total_layers = 6
        self.model_name = "distilgpt2"
        self.cache_dir = "./models"
        
        # Example multi-node configuration
        # Replace these IPs with your actual instance IPs
        self.shards = [
            ShardConfig(
                shard_id=0,
                start_layer=0,
                end_layer=2,  # layers 0, 1, 2
                host="0.0.0.0",  # Bind to all interfaces
                port=8000,
                public_ip="INSTANCE_1_PUBLIC_IP",  # Replace with actual IP
                is_input_shard=True,
                next_shard_url="http://INSTANCE_2_PUBLIC_IP:8000"  # Points to instance 2
            ),
            ShardConfig(
                shard_id=1,
                start_layer=3,
                end_layer=5,  # layers 3, 4, 5
                host="0.0.0.0",
                port=8000,  # Same port on different instance
                public_ip="INSTANCE_2_PUBLIC_IP",  # Replace with actual IP
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
                    "public_ip": shard.public_ip,
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

def generate_node_configs():
    """Generate separate config files for each node with user-provided IPs"""
    
    # Get instance IPs (you'll need to replace these)
    instance_1_ip = input("Enter Instance 1 Public IP (where shard 0 will run): ").strip()
    instance_2_ip = input("Enter Instance 2 Public IP (where shard 1 will run): ").strip()
    
    # Create full config with actual IPs
    config = MultiNodeConfig()
    config.shards[0].public_ip = instance_1_ip
    config.shards[0].next_shard_url = f"http://{instance_2_ip}:8000"
    config.shards[1].public_ip = instance_2_ip
    
    # Save complete config
    config.save_config("multi_node_config.json")
    
    # Create node-specific configs
    # Node 1 config (only shard 0)
    node1_config = MultiNodeConfig()
    node1_config.shards = [config.shards[0]]
    node1_config.save_config("node1_config.json")
    
    # Node 2 config (only shard 1)  
    node2_config = MultiNodeConfig()
    node2_config.shards = [config.shards[1]]
    node2_config.save_config("node2_config.json")
    
    print(f"\nGenerated configurations:")
    print(f"multi_node_config.json - Complete multi-node setup")
    print(f"node1_config.json - For instance 1 ({instance_1_ip}) - Shard 0")
    print(f"node2_config.json - For instance 2 ({instance_2_ip}) - Shard 1")
    
    print(f"\nDeployment Instructions:")
    print(f"1. Copy node1_config.json to Instance 1")
    print(f"2. Copy node2_config.json to Instance 2") 
    print(f"3. On Instance 1: python shard_server.py --shard-id 0 --config node1_config.json")
    print(f"4. On Instance 2: python shard_server.py --shard-id 1 --config node2_config.json")
    print(f"5. Client can connect to either instance")

def main():
    """Main function to handle configuration generation"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_node_configs()
    else:
        # Default single-node config for testing
        config = MultiNodeConfig()
        config.save_config("multi_node_config.json")
        print("Saved default multi-node configuration")
        print("Run 'python multi_node_config.py generate' to create node-specific configs")

if __name__ == "__main__":
    main() 