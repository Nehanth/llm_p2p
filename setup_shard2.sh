#!/bin/bash

echo "🚀 Setting up Shard 2 Server (Output Shard - Layers 3-5)"
echo "========================================================="

# Use known private IP addresses
INSTANCE1_IP="172.31.42.169"
INSTANCE2_IP="172.31.34.102"

echo "📝 Instance 1 (shard 1): $INSTANCE1_IP"
echo "📝 Instance 2 (this machine): $INSTANCE2_IP"

# Activate virtual environment
source venv/bin/activate

# Create shard 2 config
echo "🔧 Creating Shard 2 configuration..."
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

echo "✅ Shard 2 config created"

# Check if model exists
if [ ! -d "./models" ]; then
    echo "📥 Downloading DistilGPT-2 model..."
    python download_distillgpt2.py
fi

echo "🚀 Starting Shard 2 Server..."
echo "🔸 Layers: 3-5 (Output Shard)"
echo "🔸 Port: 8000"
echo "🔸 Receives from: http://$INSTANCE1_IP:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start shard server
python shard_server.py --shard-id 1 --config shard2_config.json 