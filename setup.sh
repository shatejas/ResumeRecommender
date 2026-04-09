#!/usr/bin/env bash
set -e

RESUME_FOLDER="${1:?Usage: ./setup.sh <resume_folder_path>}"
OLLAMA_MODEL="llama3.2"

echo "=== ATS Resume Generator Setup ==="

# 1. Install system dependencies
echo -e "\n[1/7] Checking system dependencies..."
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    brew install ollama
else
    echo "Ollama already installed."
fi

if ! brew list pango &>/dev/null 2>&1; then
    echo "Installing pango (required for PDF generation)..."
    brew install pango
else
    echo "pango already installed."
fi

# 2. Start Ollama and pull model
echo -e "\n[2/7] Starting Ollama and pulling $OLLAMA_MODEL..."
ollama serve &>/dev/null &
sleep 2
ollama pull "$OLLAMA_MODEL"

# 3. Start OpenSearch
echo -e "\n[3/7] Starting OpenSearch..."
docker-compose up -d

# 4. Wait for OpenSearch to be ready (timeout after 30s)
echo -e "\n[4/7] Waiting for OpenSearch to be ready..."
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

# 5. Install Python dependencies
echo -e "\n[5/7] Installing Python dependencies..."
pip install -r requirements.txt

# 6. Ingest resumes
echo -e "\n[6/7] Ingesting resumes from: $RESUME_FOLDER"
python ingest.py "$RESUME_FOLDER"

echo -e "\n=== Setup complete! Run 'python main.py' to generate resumes ==="
