#!/bin/bash

echo "🚀 Setting up Shard 1 Server (Input Shard - Layers 0-2)"
echo "========================================================"

# Use known private IP addresses
INSTANCE1_IP="172.31.42.169"
INSTANCE2_IP="172.31.34.102"

echo "📝 Instance 1 (this machine): $INSTANCE1_IP"
echo "📝 Instance 2 (shard 2): $INSTANCE2_IP"

# Activate virtual environment
source venv/bin/activate

# Create shard 1 config
echo "🔧 Creating Shard 1 configuration..."
cat > shard1_config.json << EOF
{
  "total_layers": 6,
  "model_name": "distilgpt2",
  "cache_dir": "./models",
  "shards": [
    {
      "shard_id": 0,
      "start_layer": 0,
      "end_layer": 2,
      "host": "0.0.0.0",
      "port": 8000,
      "public_ip": "$INSTANCE1_IP",
      "is_input_shard": true,
      "is_output_shard": false,
      "next_shard_url": "http://$INSTANCE2_IP:8000"
    }
  ]
}
EOF

echo "✅ Shard 1 config created"

# Check if model exists
if [ ! -d "./models" ]; then
    echo "📥 Downloading DistilGPT-2 model..."
    python download_distillgpt2.py
fi

echo "🚀 Starting Shard 1 Server..."
echo "🔸 Layers: 0-2 (Input Shard)"
echo "🔸 Port: 8000"
echo "🔸 Next shard: http://$INSTANCE2_IP:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start shard server
python shard_server.py --shard-id 0 --config shard1_config.json 