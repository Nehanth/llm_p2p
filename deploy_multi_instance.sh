#!/bin/bash

echo "🌐 Multi-Instance DistilGPT-2 Deployment"
echo "========================================"

show_usage() {
    echo ""
    echo "Usage: $0 <command> [instance1_ip] [instance2_ip]"
    echo ""
    echo "Commands:"
    echo "  setup     - Show setup instructions"
    echo "  test      - Test multi-instance on localhost (demo)"
    echo "  shard1    - Run shard 1 server"
    echo "  shard2    - Run shard 2 server"  
    echo "  client    - Run distributed client"
    echo "  stop      - Stop all processes"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 test"
    echo "  $0 shard1 18.220.114.100 18.220.114.140"
    echo "  $0 client 18.220.114.100 18.220.114.140"
    echo ""
}

show_setup() {
    echo ""
    echo "📋 Multi-Instance Setup Instructions:"
    echo ""
    echo "1️⃣ PREPARE INSTANCES:"
    echo "   - Instance 1: Your current instance"
    echo "   - Instance 2: 18.220.114.140 (nnarendr-decentra-2)"
    echo ""
    echo "2️⃣ SETUP INSTANCE 2:"
    echo "   ssh ubuntu@18.220.114.140"
    echo "   git clone <your-repo-url>"
    echo "   cd llm_p2p"
    echo "   ./setup_env.sh"
    echo ""
    echo "3️⃣ GET INSTANCE IPs:"
    echo "   Instance 1: curl -s http://checkip.amazonaws.com/"
    echo "   Instance 2: 18.220.114.140"
    echo ""
    echo "4️⃣ COPY SCRIPTS TO INSTANCE 2:"
    echo "   scp setup_shard2.sh ubuntu@18.220.114.140:~/llm_p2p/"
    echo "   scp shard_server.py ubuntu@18.220.114.140:~/llm_p2p/"
    echo "   scp shard_config.py ubuntu@18.220.114.140:~/llm_p2p/"
    echo ""
    echo "5️⃣ START SERVICES:"
    echo "   # On Instance 1 (this machine):"
    echo "   ./deploy_multi_instance.sh shard1 <ip1> <ip2>"
    echo ""
    echo "   # On Instance 2:"
    echo "   ./setup_shard2.sh <ip1> <ip2>"
    echo ""
    echo "   # On either instance:"
    echo "   ./deploy_multi_instance.sh client <ip1> <ip2>"
    echo ""
    echo "6️⃣ SECURITY GROUPS:"
    echo "   - Open port 8000 on both instances"
    echo "   - Open port 9000 for client access"
    echo ""
}

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

COMMAND=$1
INSTANCE1_IP=$2
INSTANCE2_IP=$3

case $COMMAND in
    "setup")
        show_setup
        ;;
        
    "test")
        echo "🧪 Testing multi-instance on localhost..."
        ./deploy_multi_node.sh demo
        ;;
        
    "shard1")
        if [ -z "$INSTANCE1_IP" ] || [ -z "$INSTANCE2_IP" ]; then
            echo "❌ Missing IP addresses"
            show_usage
            exit 1
        fi
        ./setup_shard1.sh $INSTANCE1_IP $INSTANCE2_IP
        ;;
        
    "shard2")
        if [ -z "$INSTANCE1_IP" ] || [ -z "$INSTANCE2_IP" ]; then
            echo "❌ Missing IP addresses"
            show_usage
            exit 1
        fi
        ./setup_shard2.sh $INSTANCE1_IP $INSTANCE2_IP
        ;;
        
    "client")
        if [ -z "$INSTANCE1_IP" ] || [ -z "$INSTANCE2_IP" ]; then
            echo "❌ Missing IP addresses"
            show_usage
            exit 1
        fi
        ./setup_client.sh $INSTANCE1_IP $INSTANCE2_IP
        ;;
        
    "stop")
        echo "🛑 Stopping all processes..."
        pkill -f "shard_server.py" || true
        pkill -f "distributed_client.py" || true
        ./deploy_multi_node.sh stop || true
        echo "✅ Stopped"
        ;;
        
    *)
        echo "❌ Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac 