# Distributed DistilGPT-2

Distributed LLM inference across multiple AWS instances with layer-wise sharding.

## ⚠️ IMPORTANT: Update IP Addresses First!

**Before running the system**, you MUST update the IP addresses in `config.yaml`:

### Update config.yaml (Lines 9-10)
```yaml
# Network configuration
network:
  # Set these to your actual instance IPs
  instance1_ip: "YOUR_INSTANCE_1_PRIVATE_IP"
  instance2_ip: "YOUR_INSTANCE_2_PRIVATE_IP"
```

**To find your instance IPs:**

**AWS Console**
1. Go to AWS EC2 Console
2. Click "Instances" in left sidebar
3. Select your instance
4. Copy the **Private IPv4 address** from the details panel
5. Repeat for your second instance

## Setup

### 1. Setup Environment (Both Instances)
```bash
git clone <your-repo-url>
cd llm_p2p
./setup_env.sh
```

### 2. Configure AWS Security Groups
Add inbound rule for both instances:
- **Type**: Custom TCP
- **Port**: 8000
- **Source**: 0.0.0.0/0

### 3. Start Shards

**Instance 1** (Input Shard - Layers 0-2):
```bash
./setup_shard1.sh
```

**Instance 2** (Output Shard - Layers 3-5):
```bash
./setup_shard2.sh
```

### 4. Test P2P System

**Test from ANY instance** (no master node!):
```bash
# Test input shard directly
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello P2P", "max_length": 15}'

# Check peer discovery
curl http://localhost:8000/peers

# Check health
curl http://localhost:8000/health
```

### 5. Example walkthrough

For a complete walkthrough, see the Jupyter notebook:
```bash
jupyter notebook examples/example.ipynb
```

The notebook demonstrates:
- Multi-instance setup and health checks
- Cross-instance peer discovery
- Distributed text generation
- P2P routing

## Architecture

- **Shard 0**: Layers 0-2 (Input + Embeddings)
- **Shard 1**: Layers 3-5 (Output + LM Head)
- **No Master Node**: True P2P - any server can handle requests
- **Auto-Discovery**: Shards find each other automatically
- **Direct Communication**: Shard-to-shard HTTP calls

## Requirements

- 2 AWS instances with Tesla T4 GPUs
- Ubuntu 20.04+
- Port 8000 open between instances
- 8GB+ RAM per instance
