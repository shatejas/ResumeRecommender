#!/usr/bin/env bash
set -e

OLLAMA_MODEL="mistral:7b"

echo "=== ATS Resume Generator Setup ==="

# 1. Install dependencies
echo -e "\n[1/4] Checking dependencies..."

if ! command -v brew &>/dev/null; then
    echo "ERROR: Homebrew is required. Install from https://brew.sh"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    brew install --cask docker
    echo "Starting Docker Desktop..."
    open -a Docker
    echo "Waiting for Docker to start..."
    until docker info &>/dev/null 2>&1; do sleep 2; done
    echo "Docker is ready."
else
    echo "Docker already installed."
    if ! docker info &>/dev/null 2>&1; then
        echo "Starting Docker Desktop..."
        open -a Docker
        until docker info &>/dev/null 2>&1; do sleep 2; done
    fi
fi

if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    brew install ollama
else
    echo "Ollama already installed."
fi

# 2. Start Ollama and pull models
echo -e "\n[2/4] Starting Ollama and pulling models..."
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve &>/dev/null &
sleep 2
ollama pull "$OLLAMA_MODEL"
ollama pull nomic-embed-text

# 3. Start OpenSearch
echo -e "\n[3/4] Starting OpenSearch..."
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
echo -e "\n[4/4] Installing Python dependencies..."
pip install -r requirements.txt

echo -e "\n=== Setup complete! Launching UI... ==="
python app.py
