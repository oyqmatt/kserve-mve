FROM python:3.12-slim

# PyTorch pulls in Triton, which JIT-compiles CUDA helper code at runtime and
# needs a host C compiler. The python:*-slim images do not ship one, which
# surfaces as "RuntimeError: Failed to find C compiler" on first inference.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY multi_vector_encoder.py multi_vector_encoder.py

# KServe's ModelServer listens on 8080 by default.
EXPOSE 8080

ENTRYPOINT ["python", "multi_vector_encoder.py"]
