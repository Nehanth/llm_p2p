#!/bin/bash

echo "🚀 Starting Distributed DistilGPT-2 System..."

# Activate virtual environment
source venv/bin/activate

# Generate config
echo "📝 Generating shard configuration..."
python shard_config.py

# Kill any existing processes on the ports
echo "🔍 Cleaning up existing processes..."
pkill -f "shard_server.py"
pkill -f "distributed_client.py"

# Wait a moment
sleep 2

# Start shard servers in background
echo "🔧 Starting Shard 0 (Input Shard - Layers 0-2) on port 8000..."
python shard_server.py --shard-id 0 &
SHARD0_PID=$!

echo "🔧 Starting Shard 1 (Output Shard - Layers 3-5) on port 8001..."
python shard_server.py --shard-id 1 &
SHARD1_PID=$!

# Wait for shards to start
echo "⏳ Waiting for shards to initialize..."
sleep 10

# Check if shards are healthy
echo "🩺 Checking shard health..."
curl -s http://localhost:8000/health | jq . || echo "Shard 0 not ready"
curl -s http://localhost:8001/health | jq . || echo "Shard 1 not ready"

# Start distributed client
echo "🌐 Starting Distributed Client on port 9000..."
python distributed_client.py &
CLIENT_PID=$!

# Wait for client to start
sleep 5

echo ""
echo "✅ Distributed DistilGPT-2 System Started!"
echo ""
echo "📊 System Overview:"
echo "  🔸 Shard 0 (Input):  http://localhost:8000 (Layers 0-2)"
echo "  🔸 Shard 1 (Output): http://localhost:8001 (Layers 3-5)"
echo "  🔸 Client:           http://localhost:9000 (Coordinator)"
echo ""
echo "🧪 Test Commands:"
echo "  # Check system health"
echo "  curl http://localhost:9000/health | jq ."
echo ""
echo "  # Generate text"
echo "  curl -X POST http://localhost:9000/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"The future of AI\", \"max_length\": 50}' | jq ."
echo ""
echo "📖 Documentation: http://localhost:9000/docs"
echo ""
echo "🛑 To stop the system, run: ./stop_distributed.sh"

# Save PIDs for cleanup
echo "$SHARD0_PID $SHARD1_PID $CLIENT_PID" > .distributed_pids

# Keep script running
wait 