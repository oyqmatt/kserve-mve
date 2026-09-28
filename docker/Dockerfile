FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY multi_vector_encoder.py multi_vector_encoder.py

ENTRYPOINT ["python", "multi_vector_encoder.py"]