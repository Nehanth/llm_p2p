#!/bin/bash

echo "🌐 Multi-Node DistilGPT-2 Deployment Guide"
echo "=========================================="

# Function to show deployment instructions
show_instructions() {
    echo ""
    echo "📋 Multi-Node Deployment Steps:"
    echo ""
    echo "1️⃣ SETUP INSTANCES:"
    echo "   - Launch 2 AWS/GCP instances with NVIDIA GPUs"
    echo "   - Install NVIDIA drivers on both"
    echo "   - Clone this repo on both instances"
    echo "   - Run setup on both: ./setup_env.sh"
    echo ""
    echo "2️⃣ GENERATE CONFIGS:"
    echo "   python multi_node_config.py generate"
    echo ""
    echo "3️⃣ COPY CONFIGS:"
    echo "   scp node1_config.json user@instance1:~/llm_p2p/"
    echo "   scp node2_config.json user@instance2:~/llm_p2p/"
    echo ""
    echo "4️⃣ SECURITY GROUPS:"
    echo "   - Open port 8000 on both instances"
    echo "   - Allow traffic between instances"
    echo ""
    echo "5️⃣ START SHARDS:"
    echo "   Instance 1: python shard_server.py --shard-id 0 --config node1_config.json"
    echo "   Instance 2: python shard_server.py --shard-id 1 --config node2_config.json"
    echo ""
    echo "6️⃣ START CLIENT:"
    echo "   python distributed_client.py --config multi_node_config.json"
    echo ""
    echo "🧪 TEST:"
    echo "   curl -X POST http://localhost:9000/generate \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"prompt\": \"Hello world\", \"max_length\": 20}'"
}

# Check if we're in demo mode (single instance)
if [ "$1" = "demo" ]; then
    echo "🔧 Demo Mode: Simulating multi-node on single instance"
    echo "This will use different ports to simulate multiple nodes"
    
    # Generate demo config
    python -c "
from multi_node_config import MultiNodeConfig
import json

# Create demo config with localhost IPs
config = MultiNodeConfig()
config.shards[0].public_ip = 'localhost'
config.shards[0].next_shard_url = 'http://localhost:8001'
config.shards[0].port = 8000
config.shards[1].public_ip = 'localhost'
config.shards[1].port = 8001

config.save_config('demo_multi_node.json')

# Create node configs
node1 = MultiNodeConfig()
node1.shards = [config.shards[0]]
node1.save_config('demo_node1.json')

node2 = MultiNodeConfig() 
node2.shards = [config.shards[1]]
node2.save_config('demo_node2.json')

print('✅ Demo configs generated')
"
    
    echo "🚀 Starting demo multi-node system..."
    
    # Start shards on different ports
    echo "Starting Shard 0 on port 8000..."
    python shard_server.py --shard-id 0 --config demo_node1.json &
    SHARD0_PID=$!
    
    echo "Starting Shard 1 on port 8001..."
    python shard_server.py --shard-id 1 --config demo_node2.json &
    SHARD1_PID=$!
    
    # Wait for startup
    sleep 10
    
    echo "Starting client..."
    python distributed_client.py --config demo_multi_node.json &
    CLIENT_PID=$!
    
    echo "$SHARD0_PID $SHARD1_PID $CLIENT_PID" > .demo_pids
    
    echo ""
    echo "✅ Demo Multi-Node System Started!"
    echo "🔸 Shard 0: http://localhost:8000 (Simulated Instance 1)"
    echo "🔸 Shard 1: http://localhost:8001 (Simulated Instance 2)"  
    echo "🔸 Client:  http://localhost:9000"
    echo ""
    echo "🛑 To stop: ./deploy_multi_node.sh stop"
    
elif [ "$1" = "stop" ]; then
    echo "🛑 Stopping multi-node demo..."
    
    if [ -f .demo_pids ]; then
        PIDS=$(cat .demo_pids)
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                kill $PID
            fi
        done
        rm .demo_pids
    fi
    
    pkill -f "shard_server.py"
    pkill -f "distributed_client.py"
    echo "✅ Stopped"
    
else
    show_instructions
    echo ""
    echo "💡 Quick Options:"
    echo "   ./deploy_multi_node.sh demo    # Test multi-node on single instance"
    echo "   ./deploy_multi_node.sh stop    # Stop demo"
fi 