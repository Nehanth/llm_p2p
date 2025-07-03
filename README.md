# Distributed DistilGPT-2

Distributed LLM inference across multiple AWS instances with layer-wise sharding.

## ⚠️ IMPORTANT: Update IP Addresses First!

**Before running the system**, you MUST update the hardcoded IP addresses in these files:

### 1. Update shard_server.py (Line 324-325)
```python
# Change these to your actual instance IPs:
potential_peers = [
    {"host": "YOUR_INSTANCE_1_PRIVATE_IP", "port": 8000},  # Instance 1
    {"host": "YOUR_INSTANCE_2_PRIVATE_IP", "port": 8000},  # Instance 2
]
```

### 2. Update setup_shard1.sh (Lines 6-7)
```bash
# Change these to your actual instance IPs:
INSTANCE1_IP="YOUR_INSTANCE_1_PRIVATE_IP"
INSTANCE2_IP="YOUR_INSTANCE_2_PRIVATE_IP"
```

### 3. Update setup_shard2.sh (Lines 6-7)
```bash
# Change these to your actual instance IPs:
INSTANCE1_IP="YOUR_INSTANCE_1_PRIVATE_IP"
INSTANCE2_IP="YOUR_INSTANCE_2_PRIVATE_IP"
```

**To find your instance IPs:**

**Option 1: AWS Console (Easiest)**
1. Go to AWS EC2 Console
2. Click "Instances" in left sidebar
3. Select your instance
4. Copy the **Private IPv4 address** from the details panel
5. Repeat for your second instance

**Option 2: Command Line**
```bash
# Get private IP
curl -s http://169.254.169.254/latest/meta-data/local-ipv4

# Or use hostname command
hostname -I | awk '{print $1}'
```

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
