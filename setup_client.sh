#!/bin/bash

echo "🌐 Setting up Distributed Client"
echo "================================="

# Get configuration
INSTANCE1_IP=${1:-"INSTANCE1_IP"}
INSTANCE2_IP=${2:-"INSTANCE2_IP"}

if [ "$INSTANCE1_IP" = "INSTANCE1_IP" ] || [ "$INSTANCE2_IP" = "INSTANCE2_IP" ]; then
    echo "Usage: $0 <instance1_ip> <instance2_ip>"
    echo "Example: $0 18.220.114.100 18.220.114.140"
    exit 1
fi

echo "📝 Instance 1 (Shard 1): $INSTANCE1_IP"
echo "📝 Instance 2 (Shard 2): $INSTANCE2_IP"

# Activate virtual environment
source venv/bin/activate

# Create client config
echo "🔧 Creating client configuration..."
cat > client_config.json << EOF
{
  "total_layers": 6,
  "model_name": "distilgpt2",
  "cache_dir": "./models",
  "shards": [
    {
      "shard_id": 0,
      "start_layer": 0,
      "end_layer": 2,
      "host": "$INSTANCE1_IP",
      "port": 8000,
      "public_ip": "$INSTANCE1_IP",
      "is_input_shard": true,
      "is_output_shard": false,
      "next_shard_url": "http://$INSTANCE2_IP:8000"
    },
    {
      "shard_id": 1,
      "start_layer": 3,
      "end_layer": 5,
      "host": "$INSTANCE2_IP",
      "port": 8000,
      "public_ip": "$INSTANCE2_IP",
      "is_input_shard": false,
      "is_output_shard": true,
      "next_shard_url": null
    }
  ]
}
EOF

echo "✅ Client config created"

echo "🚀 Starting Distributed Client..."
echo "🔸 Managing 2 shards across instances"
echo "🔸 API: http://localhost:9000"
echo "🔸 Docs: http://localhost:9000/docs"
echo ""
echo "🧪 Test with:"
echo "curl -X POST http://localhost:9000/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"prompt\": \"Hello world\", \"max_length\": 20}'"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start client
python distributed_client.py --config client_config.json 