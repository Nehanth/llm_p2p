#!/bin/bash

echo "🚀 Complete Setup for Distributed DistilGPT-2"
echo "=============================================="








# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root. Run as ubuntu user."
   exit 1
fi

# System updates
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
sudo apt install -y curl wget git build-essential software-properties-common

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Installing Python3..."
    sudo apt install -y python3 python3-pip python3-venv python3-dev
else
    echo "Python3 already installed"
fi

# Install additional system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3.12-venv \
    python3-pip \
    nvidia-utils-535 \
    ubuntu-drivers-common \
    linux-headers-$(uname -r) \
    jq \
    htop \
    net-tools

# Detect GPU and install NVIDIA drivers
echo "Detecting GPU and installing NVIDIA drivers..."

# Check if NVIDIA GPU exists
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected!"
    
    # Check if nvidia-smi works
    if nvidia-smi &> /dev/null; then
        echo "NVIDIA drivers already working!"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "Installing NVIDIA drivers..."
        
        # Install recommended driver
        sudo ubuntu-drivers autoinstall
        
        # Also install specific driver version
        sudo apt install -y nvidia-driver-535 nvidia-utils-535
        
        echo "NVIDIA drivers installed. REBOOT REQUIRED!"
        echo ""
        echo "Please run 'sudo reboot' and then run this script again."
        echo "   After reboot, run: ./setup_env.sh"
        exit 0
    fi
else
    echo "No NVIDIA GPU detected. Continuing with CPU-only setup."
fi

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed"
else
    echo "requirements.txt not found!"
    exit 1
fi

# Download model if not exists
echo "Checking DistilGPT-2 model..."
if [ ! -d "models" ]; then
    echo "Downloading DistilGPT-2 model..."
    python scripts/download_distillgpt2.py
    echo "Model downloaded"
else
    echo "Model already exists"
fi

# Test GPU setup
echo "Testing GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Running on CPU')
"

# Security group reminders
echo ""
echo "Security Group Configuration:"
print_warning "Make sure these ports are open in your AWS Security Group:"
echo "   - Port 8000: Shard servers"
echo "   - Port 9000: Client API"
echo "   - Port 22: SSH access"

# Get instance IP
echo ""
echo "Instance Information:"
INSTANCE_IP=$(curl -s http://checkip.amazonaws.com/ || echo "Unable to detect IP")
echo "   Public IP: $INSTANCE_IP"
echo "   Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo 'Not available')"

echo ""
print_status "🎉 Setup complete!"
echo ""
echo "Next Steps:"
echo "   1. If you saw 'REBOOT REQUIRED' above, reboot now: sudo reboot"
echo "   2. Test the setup: ./deploy_multi_node.sh demo"
echo "   3. For multi-instance: Use your IP ($INSTANCE_IP)"
echo ""
echo "Commands:"
echo "   ./setup_shard1.sh <ip1> <ip2>  # Start shard 1"
echo "   ./setup_shard2.sh <ip1> <ip2>  # Start shard 2"  
echo "   ./setup_client.sh <ip1> <ip2>  # Start client"
echo ""
echo "Documentation: Check README.md for detailed instructions" 