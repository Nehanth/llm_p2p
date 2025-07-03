#!/bin/bash

echo "🛑 Stopping Distributed DistilGPT-2 System..."

# Kill processes by PID if file exists
if [ -f .distributed_pids ]; then
    echo "📝 Reading process IDs..."
    PIDS=$(cat .distributed_pids)
    for PID in $PIDS; do
        if kill -0 $PID 2>/dev/null; then
            echo "⏹️  Stopping process $PID..."
            kill $PID
        fi
    done
    rm .distributed_pids
fi

# Kill any remaining processes by name
echo "🔍 Cleaning up remaining processes..."
pkill -f "shard_server.py"
pkill -f "distributed_client.py"

# Wait a moment for cleanup
sleep 2

echo "✅ All processes stopped!"

# Check if ports are free
echo "🔍 Checking ports..."
if ! netstat -tuln | grep -q ":8000 "; then
    echo "  ✅ Port 8000 is free"
else
    echo "  ⚠️  Port 8000 still in use"
fi

if ! netstat -tuln | grep -q ":8001 "; then
    echo "  ✅ Port 8001 is free"
else
    echo "  ⚠️  Port 8001 still in use"
fi

if ! netstat -tuln | grep -q ":9000 "; then
    echo "  ✅ Port 9000 is free"
else
    echo "  ⚠️  Port 9000 still in use"
fi 