#!/usr/bin/env bash
set -e

OLLAMA_MODEL="mistral:7b"

echo "=== ATS Resume Generator Setup ==="

# 1. Check dependencies
echo -e "\n[1/5] Checking dependencies..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker Desktop first."
    exit 1
fi
if ! docker info &>/dev/null 2>&1; then
    echo "Starting Docker Desktop..."
    open -a Docker
    until docker info &>/dev/null 2>&1; do sleep 2; done
fi
echo "Docker is ready."

if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    exit 1
fi
echo "Python already installed."

# 2. Start Ollama and pull models
echo -e "\n[2/5] Starting Ollama and pulling models..."
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve &>/dev/null &
sleep 2
ollama pull "$OLLAMA_MODEL"
ollama pull nomic-embed-text

# 3. Start OpenSearch
echo -e "\n[3/5] Starting OpenSearch..."
docker-compose up -d

echo "Waiting for OpenSearch to be ready..."
retries=0
max_retries=15
until curl -s http://localhost:9200 >/dev/null 2>&1; do
    retries=$((retries + 1))
    if [ $retries -ge $max_retries ]; then
        echo "ERROR: OpenSearch failed to start. Check logs with: docker logs opensearch"
        exit 1
    fi
    sleep 2
done
echo "OpenSearch is up!"

# 4. Install Python dependencies
echo -e "\n[4/5] Installing Python dependencies..."
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium

# 5. Launch UI
echo -e "\n[5/5] Launching UI..."
python3 app.py
