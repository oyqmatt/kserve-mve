# ColQwen Multi-Vector Encoder

A [ColQwen](https://github.com/illuin-tech/colpali)-style **multi-vector (late-interaction)** encoder served with [KServe](https://kserve.github.io/website/). Queries and documents are encoded into per-token embedding matrices, and scored with a **MaxSim** late-interaction similarity, the same retrieval paradigm used by ColBERT/ColPali for text and visual document retrieval.

## How it works

The encoder produces one embedding vector per token/visual patch rather than a single pooled vector:

```text
Query 0 shape:    (61, 128)
Document 0 shape: (1034, 128)
```

`model.similarity(...)` then computes a MaxSim score matrix (rows = queries, columns = documents):

```text
tensor([[53.5625, 49.2036, 46.6958, 45.4949],
        [45.6436, 53.1328, 45.0957, 45.5176]])
```

- `encode_query(list[str])` / `encode_document(list[str | image])` → per-item embedding tensors.
- `similarity(query_embeddings, document_embeddings)` → score tensor.

## Repository layout

```text
.
├── multi_vector_encoder.py   # KServe Model wrapper + HTTP server (primary entrypoint)
├── serve.py                  # Standalone local demo (vidore/colqwen-omni-v0.1)
├── requirements.txt          # Python dependencies (currently unpinned)
├── docker/Dockerfile         # Container image for the serving entrypoint
├── Makefile                  # `make build` shortcut
├── test.sh / test.json       # Curl smoke test + sample payload
└── .devcontainer/            # GPU devcontainer (docker-in-docker, Python 3.12)
```

Two entrypoints exist:

- `multi_vector_encoder.py` — the production serving path. Subclasses `kserve.Model`, loads `mixedbread-ai/mxbai-edge-colbert-v0-17m`, and exposes the KServe predict endpoint.
- `serve.py` — a scripted local demo using `vidore/colqwen-omni-v0.1` against example document images; useful for inspecting tensor shapes and scores without a server.

## Prerequisites

- Python 3.12
- NVIDIA GPU + drivers (the `.devcontainer` requests `--gpus all`)
- Docker (for container builds)

## Quickstart

### Local / devcontainer

```bash
pip install -r requirements.txt
python multi_vector_encoder.py
```

The server listens on port `8080` by default. For a GPU-backed environment, open the repo in the provided devcontainer (it enables docker-in-docker and passes through all GPUs).

### Smoke test

With the server running:

```bash
./test.sh
# equivalent to:
# curl -H "Content-Type: application/json" \
#   http://localhost:8080/v1/models/mxbai:predict -d @./test.json
```

`test.json`:

```json
{
  "queries": ["Which planet is known as the Red Planet?"],
  "documents": [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
  ]
}
```

## API reference

**Endpoint:** `POST /v1/models/mxbai:predict`

**Request body**

| Field       | Type            | Description                                  |
| ----------- | --------------- | -------------------------------------------- |
| `queries`   | `string[]`      | Query texts to encode.                       |
| `documents` | `string[]`      | Document texts (or image paths/URLs) to score.|

**Response body**

```json
{ "predictions": [[53.56, 49.20, 46.70, 45.49]] }
```

`predictions` is a 2D array: one row per query, one column per document.

## Docker

```bash
make build
# equivalent to:
# docker build -f docker/Dockerfile -t localhost/kserve_img:latest .
```

Run the image (GPU recommended):

```bash
docker run --gpus all -p 8080:8080 localhost/kserve_img:latest
```

> The current image installs dependencies and starts the server, but does **not** bake model weights into the image — the model is downloaded from Hugging Face on first start. See the next steps below.

## Next steps for production deployment

The repository currently targets local development. The following work is required before production use.

### 1. Dependency and model reproducibility

- [ ] Pin versions in `requirements.txt` (or adopt a lock file such as `uv.lock` / `pip-compile`). The `sentence_transformers.MultiVectorEncoder` API requires a specific `sentence-transformers` release; unpinned installs are not reproducible.
- [ ] Pin/record the exact model revision (`mixedbread-ai/mxbai-edge-colbert-v0-17m`) instead of tracking the default branch.
- [ ] Bake weights into the image or mount a pre-populated model cache volume to eliminate cold-start downloads and external network dependency.

### 2. Container hardening

- [ ] Switch to a CUDA-enabled base image (e.g. `nvidia/cuda` + matching PyTorch) so GPU serving does not depend on the host Python environment.
- [ ] Run the container as a non-root user.
- [ ] Add a `.dockerignore` to keep the build context small and avoid leaking local files.
- [ ] Add a `HEALTHCHECK` hitting the model readiness endpoint.
- [ ] Consider a multi-stage build to reduce final image size.

### 3. Serving and scaling

- [ ] Add a KServe `InferenceService` manifest (or Knative/RawDeployment) instead of running the container directly.
- [ ] Configure GPU node scheduling, resource requests/limits, and `minReplicas` for warm capacity.
- [ ] Enable autoscaling and request batching.
- [ ] Define readiness/liveness probes against the KServe ready endpoint.

### 4. Configuration

- [ ] Make the model path and served model name configurable via arguments/env vars instead of hardcoded values in `multi_vector_encoder.py`.
- [ ] Remove unused imports (`cffi`, `torch`, `PIL.Image`, `kserve.model_server.app`, `generate_uuid`).
- [ ] Support image/path inputs for `documents` (currently `serve.py` demonstrates image URLs, but the server path is text-only).

### 5. Observability

- [ ] Structured JSON logging with request IDs.
- [ ] Metrics for request latency, throughput, batch size, and GPU utilization.
- [ ] Distributed tracing across the gateway and model server.

### 6. Security

- [ ] Terminate TLS and enable authentication at the ingress/gateway.
- [ ] Apply network policies and least-privilege service accounts.
- [ ] Scan images for CVEs and never bake secrets/credentials into the image.

### 7. CI/CD

- [ ] Add lint/typecheck/test jobs.
- [ ] Build and push the image to a registry on tagged releases.
- [ ] Deploy via GitOps with a documented rollback/versioning strategy.

### 8. Testing

- [ ] Unit tests for request parsing and score shaping.
- [ ] Integration tests against a running server using `test.json`.
- [ ] Load tests to validate latency under batch/concurrency.

### 9. Input robustness

- [ ] Handle remote image fetching failures, oversized documents, and malformed payloads gracefully.
- [ ] Validate and cap input sizes to protect GPU memory.
