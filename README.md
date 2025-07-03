# Distributed DistilGPT-2

Distributed LLM inference across multiple AWS instances with layer-wise sharding.

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

**Instance 1** (172.31.42.169):
```bash
./setup_shard1.sh
```

**Instance 2** (172.31.34.102):
```bash
./setup_shard2.sh
```

### 4. Start Client

**On Instance 1**:
```bash
./setup_client.sh
```

### 5. Test

```bash
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_length": 20}'
```

## Requirements

- 2 AWS instances with Tesla T4 GPUs
- Ubuntu 20.04+
- Port 8000 open between instances
- 8GB+ RAM per instance
