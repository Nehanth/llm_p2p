# 🌐 Distributed DistilGPT-2

**Multi-instance distributed LLM inference** with layer-wise sharding across GPU instances.

## ✨ Features

- 🚀 **GPU-accelerated** inference on Tesla T4/L4 
- 🔄 **Layer sharding** - split model across instances
- 🌐 **Multi-node** - true distributed serving
- ⚡ **FastAPI** - REST API with auto-docs
- 🧪 **Autoregressive** generation with sampling

## 🏗️ Architecture

```
Instance 1 (Shard 0)    Instance 2 (Shard 1)
┌─────────────────┐    ┌─────────────────┐
│  Layers 0-2     │───▶│  Layers 3-5     │
│  (Input Shard)  │    │  (Output Shard) │
└─────────────────┘    └─────────────────┘
        ▲                        │
        │        Client          │
        └────────────────────────┘
```

## 🚀 Quick Start

### 1️⃣ Setup Both Instances

```bash
# Clone repo
git clone <your-repo-url>
cd llm_p2p

# Setup environment (installs NVIDIA drivers if needed)
./setup_env.sh

# Reboot if NVIDIA drivers were installed
sudo reboot
```

### 2️⃣ Get Instance IPs

```bash
# Get your public IP
curl -s http://checkip.amazonaws.com/
```

### 3️⃣ Start Shards

**Instance 1** (Input Shard):
```bash
./setup_shard1.sh <instance1_ip> <instance2_ip>
```

**Instance 2** (Output Shard):  
```bash
./setup_shard2.sh <instance1_ip> <instance2_ip>
```

### 4️⃣ Start Client

**On either instance**:
```bash
./setup_client.sh <instance1_ip> <instance2_ip>
```

### 5️⃣ Test

```bash
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_length": 30}'
```

## 🧪 Local Testing

Test multi-instance on single machine:
```bash
./deploy_multi_node.sh demo
```

## 📊 API Endpoints

- **Health**: `GET /health` - Check system status
- **Generate**: `POST /generate` - Text generation
- **Docs**: `GET /docs` - Interactive API docs
- **Shards**: `GET /shards` - Shard information

## ⚙️ Configuration

Each script auto-generates configs with your IPs:
- `shard1_config.json` - Shard 0 configuration  
- `shard2_config.json` - Shard 1 configuration
- `client_config.json` - Client coordination

## 🔧 Requirements

- **GPU**: NVIDIA T4/L4 with 16GB+ VRAM
- **OS**: Ubuntu 20.04+ with NVIDIA drivers
- **Network**: Open port 8000 (shards) and 9000 (client)
- **Memory**: 8GB+ RAM per instance

## 🐛 Troubleshooting

**No GPU detected**:
```bash
nvidia-smi  # Should show your GPU
sudo reboot  # If drivers just installed
```

**Connection refused**:
- Check security groups (open ports 8000, 9000)
- Verify IPs are correct
- Ensure both shards are running

**Slow generation**:
- Check `nvidia-smi` for GPU utilization  
- Network latency affects cross-shard communication
- Use lower `max_length` for testing

## 🌟 What's Next?

This is a foundation for **P2P distributed LLMs**. Next steps:
- IPFS model distribution
- DHT-based node discovery  
- Decentralized request routing
- Multi-shard redundancy

## 📝 Files

- `setup_env.sh` - Environment setup
- `setup_shard1.sh` - Start input shard  
- `setup_shard2.sh` - Start output shard
- `setup_client.sh` - Start coordination client
- `shard_server.py` - Shard server implementation
- `distributed_client.py` - Client implementation 