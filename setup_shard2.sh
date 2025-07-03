#!/bin/bash

echo "🌐 Setting up P2P Shard 1 (Output Peer - Layers 3-5)"
echo "===================================================="

# Use known private IP addresses
INSTANCE1_IP="172.31.42.169"
INSTANCE2_IP="172.31.34.102"

echo "📝 Instance 1 (peer 1): $INSTANCE1_IP"
echo "📝 Instance 2 (this peer): $INSTANCE2_IP"

# Activate virtual environment
source venv/bin/activate

# Create shard 2 config
echo "🔧 Creating P2P Shard 1 configuration..."
cat > shard2_config.json << EOF
{
  "total_layers": 6,
  "model_name": "distilgpt2",
  "cache_dir": "./models",
  "shards": [
    {
      "shard_id": 1,
      "start_layer": 3,
      "end_layer": 5,
      "host": "0.0.0.0",
      "port": 8000,
      "public_ip": "$INSTANCE2_IP",
      "is_input_shard": false,
      "is_output_shard": true,
      "next_shard_url": null
    }
  ]
}
EOF

echo "✅ P2P Shard 1 config created"

# Check if model exists
if [ ! -d "./models" ]; then
    echo "📥 Downloading DistilGPT-2 model..."
    python download_distillgpt2.py
fi

echo "🚀 Starting P2P Shard 1 (Output Peer)..."
echo "🔸 Layers: 3-5 (Output Generation)"
echo "🔸 Port: 8000"
echo "🔸 P2P Partner: http://$INSTANCE1_IP:8000"
echo ""
echo "🌟 P2P Endpoints Available:"
echo "   - Direct Generation: POST /generate (auto-routes to input peer)"
echo "   - Peer Discovery: GET /peers"
echo "   - Health Check: GET /health"
echo "   - Inter-shard: POST /process"
echo ""
echo "🧪 Test P2P Generation:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"Hello P2P\", \"max_length\": 15}'"
echo ""
echo "🌐 No coordinator needed - this is a true P2P peer!"
echo "🔀 Requests to this peer auto-route to input peer"
echo "Press Ctrl+C to stop"
echo ""

# Start shard server
python shard_server.py --shard-id 1 --config shard2_config.json 