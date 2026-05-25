#!/usr/bin/env bash

# Color codes for clean output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}   🛠️  Antigravity Local RAG - Setup Script     ${NC}"
echo -e "${BLUE}===============================================${NC}"

# 1. Check Python Version
echo -e "\n${BLUE}[1/6] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 could not be found. Please install Python 3.10+ and try again.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found.${NC}"

# 2. Setup Virtual Environment
echo -e "\n${BLUE}[2/6] Configuring Python virtual environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating new virtual environment in .venv...${NC}"
    python3 -m venv .venv
fi
source .venv/bin/activate
echo -e "${GREEN}✓ Activated virtual environment.${NC}"

# 3. Install Dependencies
echo -e "\n${BLUE}[3/6] Installing dependencies from requirements.txt...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install Python packages. Check your internet connection or requirements.txt.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python packages installed successfully.${NC}"

# 4. Pre-download Embedding Model
echo -e "\n${BLUE}[4/6] Caching HuggingFace SentenceTransformer model locally...${NC}"
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to download BAAI/bge-small-en-v1.5 embedding model.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Embedding model successfully cached.${NC}"

# 5. Check and configure Ollama LLM
echo -e "\n${BLUE}[5/6] Inspecting Ollama local service...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️ ollama CLI is not installed locally.${NC}"
    echo -e "${YELLOW}If you plan to run Ollama inside Docker, make sure you configure OLLAMA_HOST accordingly.${NC}"
    echo -e "${YELLOW}Otherwise, please download Ollama from https://ollama.com${NC}"
else
    echo -e "${GREEN}✓ ollama CLI found.${NC}"
    
    # Check if Ollama service is running
    echo "Checking if Ollama daemon is running..."
    curl -s -f http://localhost:11434/api/tags > /dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️ Ollama service is not running. Attempting to launch it (macOS)...${NC}"
        open -a Ollama &> /dev/null || ollama serve &> /dev/null &
        sleep 5
    fi

    # Pull Gemma model
    echo -e "${BLUE}Pulling local LLM model (gemma4:e4b) via Ollama...${NC}"
    ollama pull gemma4:e4b
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model gemma4:e4b is downloaded and ready.${NC}"
    else
        echo -e "${YELLOW}⚠️ Failed to pull gemma4:e4b automatically. You can download it manually with: 'ollama pull gemma4:e4b'${NC}"
    fi
fi

# 6. Verify System with Test Suite
echo -e "\n${BLUE}[6/6] Running system verification & grounding tests...${NC}"
python3 test_rag.py
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}===============================================${NC}"
    echo -e "${GREEN}🎉 SETUP COMPLETED SUCCESSFULLY!                ${NC}"
    echo -e "${GREEN}===============================================${NC}"
    echo -e "Your offline local RAG system is fully working."
    echo -e "To start the FastMCP Server, run:"
    echo -e "  ${BLUE}source .venv/bin/activate && python src/server.py${NC}"
    echo -e "==============================================="
else
    echo -e "${RED}❌ System verification tests returned non-zero status. Please inspect details above.${NC}"
    exit 1
fi
