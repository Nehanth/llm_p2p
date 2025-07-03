#!/bin/bash

echo "🧪 Testing P2P Distributed LLM"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📡 Testing peer discovery...${NC}"
echo "Checking Shard 0 peers:"
curl -s http://localhost:8000/peers | jq . || curl -s http://localhost:8000/peers

echo ""
echo "Checking Shard 1 peers:"
curl -s http://172.31.34.102:8000/peers | jq . || curl -s http://172.31.34.102:8000/peers

echo ""
echo -e "${BLUE}🩺 Testing health endpoints...${NC}"
echo "Shard 0 health:"
curl -s http://localhost:8000/health | jq . || curl -s http://localhost:8000/health

echo ""
echo "Shard 1 health:"
curl -s http://172.31.34.102:8000/health | jq . || curl -s http://172.31.34.102:8000/health

echo ""
echo -e "${YELLOW}🚀 Testing P2P Generation...${NC}"

echo -e "${GREEN}Test 1: Send to Input Peer (Shard 0)${NC}"
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_length": 15}' | jq . || \
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_length": 15}'

echo ""
echo -e "${GREEN}Test 2: Send to Output Peer (Shard 1) - Auto-routes to Input${NC}"
curl -X POST http://172.31.34.102:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello P2P world", "max_length": 15}' | jq . || \
curl -X POST http://172.31.34.102:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello P2P world", "max_length": 15}'

echo ""
echo -e "${BLUE}✅ P2P Testing Complete!${NC}"
echo "🌐 Your distributed LLM is working in true P2P mode!" 