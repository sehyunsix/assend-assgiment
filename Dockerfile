FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Set Python path to include src/analysis
ENV PYTHONPATH="${PYTHONPATH}:/app/src/analysis"

# Create output and data directories
RUN mkdir -p /output /data

# Default output directory environment variable
ENV OUTPUT_DIR=/output

# Entrypoint script to handle historical/realtime commands
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
if [ "$1" == "historical" ]; then
    SUBDIR=${2:-""}
    OUTPUT_PATH="/output/historical"
    if [ -n "$SUBDIR" ]; then
        OUTPUT_PATH="/output/historical/$SUBDIR"
    fi
    mkdir -p "$OUTPUT_PATH"
    python src/analysis/engine_runner.py historical --data /data --output "$OUTPUT_PATH"
elif [ "$1" == "realtime" ]; then
    mkdir -p /output/realtime
    python src/analysis/engine_runner.py realtime --output /output/realtime
elif [ "$1" == "scorecard" ]; then
    OUTPUT_PATH="/output/historical"
    if [ -n "$2" ]; then
        OUTPUT_PATH="/output/historical/$2"
    fi
    python src/research/scorecard.py "$OUTPUT_PATH"
else
    echo "Usage: docker run your-image [historical [subdir]|realtime|scorecard [subdir]]"
    exit 1
fi
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
