#!/usr/bin/env bash
set -e

RESUME_FOLDER="${1:?Usage: ./setup.sh <resume_folder_path>}"
OLLAMA_MODEL="mistral:7b"

echo "=== ATS Resume Generator Setup ==="

# 1. Install Ollama if not present
echo -e "\n[1/5] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    brew install ollama
else
    echo "Ollama already installed."
fi

# 2. Start Ollama and pull models
echo -e "\n[2/5] Starting Ollama and pulling models..."
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve &>/dev/null &
sleep 2
ollama pull "$OLLAMA_MODEL"
ollama pull nomic-embed-text

# 3. Start OpenSearch
# 3. Start OpenSearch
echo -e "\n[3/5] Starting OpenSearch..."
docker-compose up -d

echo -e "Waiting for OpenSearch to be ready..."
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
pip install -r requirements.txt

# 5. Ingest resumes
echo -e "\n[5/5] Ingesting resumes from: $RESUME_FOLDER"
python ingest.py "$RESUME_FOLDER"

echo -e "\n=== Setup complete! Run 'python main.py' to generate resumes ==="
