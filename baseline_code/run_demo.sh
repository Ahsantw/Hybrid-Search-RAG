#!/bin/bash

# Optional: Activate virtual environment
# source /path/to/venv/bin/activate

echo "Downloading and Converting llama Model to OpenVino IR INT4"
if python src/convert_llama_to_open.py; then
    echo "✅ Downloaded and Converted llama Model to OpenVino IR INT4 sucessfully."
else
    echo "❌ Error occurred in cDownloading and Converting llama Model to OpenVino IR INT4. Please check logs."
    exit 1
fi

echo "Saving PDF's embedings into db"
if python src/vector_db.py; then
    echo "✅ Saving PDF's embedings into db executed successfully."
else
    echo "❌ Error occurred while saving PDF's embedings into db. Please check logs"
    exit 1
fi

# --- Run rag_cli.py ---
echo "Running Inference..."
if python rag_cli.py; then
    echo "✅ rag_cli.py executed successfully."
else
    echo "❌ Error occurred in during Inference. Please check logs"
    exit 1
fi