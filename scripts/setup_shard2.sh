#!/bin/bash

echo "🌐 Setting up P2P Shard 1 (Output Peer - Layers 3-5)"
echo "===================================================="

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -d "./models" ]; then
    echo "Downloading DistilGPT-2 model..."
    python scripts/download_distillgpt2.py
fi

echo "🚀 Starting P2P Shard 1 (Output Peer)..."
echo "Layers: 3-5 (Output Generation)"
echo "Port: 8000"
echo ""
echo "P2P Endpoints Available:"
echo "   - Direct Generation: POST /generate (auto-routes to input peer)"
echo "   - Peer Discovery: GET /peers"
echo "   - Health Check: GET /health"
echo "   - Inter-shard: POST /process"
echo ""
echo "Test P2P Generation:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"Hello P2P\", \"max_length\": 15}'"
echo ""
echo "Requests to this peer auto-route to input peer"
echo "Press Ctrl+C to stop"
echo ""

# Start shard server
python main.py --shard-id 1 --config config.yaml 