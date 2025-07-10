#!/usr/bin/env python3
"""
Main entry point for distributed DistilGPT-2 inference
"""

import argparse
import uvicorn
from fastapi import FastAPI

from config_loader import Config, ShardConfig
from sharded_model import ModelLoader
from p2p_server import P2PServer
from api_routes import APIRoutes
from utils import get_logger

logger = get_logger(__name__)

class ShardServer:
    """Main shard server class - orchestrates all components"""
    
    def __init__(self, shard_config: ShardConfig, model_config: Config):
        self.shard_config = shard_config
        self.model_config = model_config
        
        # Initialize components
        self.model_loader = ModelLoader(shard_config, model_config)
        
        # Load model and tokenizer
        self.model = self.model_loader.load_model()
        self.tokenizer = self.model_loader.load_tokenizer()
        
        # Initialize P2P server
        self.p2p_server = P2PServer(shard_config, self.model, self.tokenizer)
        
        # Setup API routes
        self.app = FastAPI(
            title=f"P2P Shard {shard_config.shard_id} Server",
            description=f"P2P Distributed DistilGPT-2 Shard (Layers {shard_config.start_layer}-{shard_config.end_layer})"
        )
        
        self.api_routes = APIRoutes(shard_config, self.model, self.tokenizer, self.p2p_server)
        self.api_routes.setup_routes(self.app)
        
    def run(self):
        """Start the shard server"""
        print(f"Starting P2P Shard {self.shard_config.shard_id} Server...")
        print(f"Layers: {self.shard_config.start_layer}-{self.shard_config.end_layer}")
        print(f"Server: http://{self.shard_config.host}:{self.shard_config.port}")
        print(f"Device: {self.model_loader.device}")
        print(f"P2P Endpoints:")
        print(f"   - Generate: POST /generate")
        print(f"   - Peers: GET /peers")
        print(f"   - Health: GET /health")
        
        uvicorn.run(
            self.app,
            host=self.shard_config.host,
            port=self.shard_config.port,
            log_level="info"
        )

def main():
    """Main function to start the shard server"""
    parser = argparse.ArgumentParser(description="Run a model shard server")
    parser.add_argument("--shard-id", type=int, required=True, help="Shard ID to run")
    parser.add_argument("--config", type=str, default="shard_config.json", help="Config file path")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and args.config.endswith('.yaml'):
        config = Config(args.config)
    else:
        config = Config()
    
    shard_config = config.get_shard_config(args.shard_id)
    
    # Create and run server
    server = ShardServer(shard_config, config)
    server.run()

if __name__ == "__main__":
    main() 