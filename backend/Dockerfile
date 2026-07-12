# Python 3.11 slim image (master prompt requires 3.11+)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install CPU-only PyTorch FIRST ─────────────────────────────
# torch's default build bundles ~2GB of NVIDIA CUDA libs that we don't need
# (we run on CPU for embeddings). Installing from the CPU wheel index first
# means sentence-transformers finds torch already satisfied and skips CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# uvicorn with --reload enables hot-reload when mounted code changes
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
