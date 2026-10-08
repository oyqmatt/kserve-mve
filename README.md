# ColQwen Omni Embedding API

A [ColQwen Omni](https://huggingface.co/vidore/colqwen-omni-v0.1) multi-vector encoder served with [KServe](https://kserve.github.io/website/). The service exposes separate embedding endpoints for audio documents, text documents, and queries. Query inputs can be text or audio. It returns token-level embeddings, and the KServe V2 `infer` endpoint scores precomputed query and document embeddings with ColBERT **MaxSim** late interaction.

## How it works

The encoder produces one vector per token or audio feature rather than a single pooled vector. Responses return the full batch unchanged: a list of variable-length matrices shaped `(tokens, dimensions)`, even when the request contains a single input.

```text
Query 0 shape:    (61, 128)
Document 0 shape: (1034, 128)
```

- `encode_document(...)` embeds audio or text documents.
- `encode_query(...)` embeds text or audio queries.
- The KServe V2 `infer` endpoint computes a MaxSim score matrix from previously
  returned query and document embeddings.

```text
Query tokens:    (61, 128)
Document tokens: (1034, 128)
MaxSim score:    53.5625
```

## Repository layout

```text
.
├── multi_vector_encoder.py   # KServe model wrapper + embed + MaxSim endpoints
├── requirements.txt          # Python dependencies (currently unpinned)
├── docker/Dockerfile         # Container image for the serving entrypoint
├── charts/colqwen-mve/       # Helm chart: ServingRuntime + InferenceService (serverless KServe)
├── Makefile                  # `make build` shortcut
├── test.py / test.json       # Stdlib smoke test (prints shapes) + payload for Helm NOTES
└── .devcontainer/            # Devcontainer (docker-in-docker, Python 3.12)
```

`multi_vector_encoder.py` loads `vidore/colqwen-omni-v0.1` and serves the three REST embedding endpoints below, alongside KServe health endpoints. Inference runs on **CPU** by default: the 4.4B checkpoint does not fit a consumer GPU, so plan for roughly 9 GiB of RAM for the weights alone plus headroom for activations.

## Prerequisites

- Python 3.12
- ~16 GiB RAM or more (the checkpoint is 4.4B parameters and runs on CPU)
- Docker (for container builds)

## Quickstart

### Local / devcontainer

```bash
pip install -r requirements.txt
python multi_vector_encoder.py
```

The server listens on port `8080` by default. The first start downloads the model from Hugging Face; subsequent starts reuse the cache.

> CPU inference of a 4.4B model is slow — expect several seconds per request even on a many-core host. Set `DEVICE=cuda` to use a GPU if one has enough VRAM.

### Smoke test

With the server running:

```bash
python test.py
```

The script uses only the standard library. It posts documents and a text query
to `/embed/docs` and `/embed/query`, synthesizes a PCM16 sine for
`/embed/audio`, then runs the full round-trip through the MaxSim `infer`
endpoint, printing each returned embedding shape, e.g.:

```text
embed/docs   shapes: [[15, 128], [17, 128]]
embed/query  shapes: [[10, 128]]
embed/audio  shapes: [[23, 128]]
maxsim       shape: [1, 2]
```

Useful flags: `--base-url`, `--skip-audio`, `--skip-maxsim`.

> **For API consumers:** the precise client-facing contract — request/response
> schemas, audio formats, error semantics, guarantees and limits — lives in
> [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md). This README section is a
> summary.

## API reference

The server exposes three JSON endpoints:

### `POST /embed/audio`

Embed one audio document. `audio` is base64-encoded, already-decoded **raw little-endian PCM16, 16-bit mono at 16 kHz**; it is converted to float32 in `[-1, 1]` without further resampling.

```json
{"audio": "<base64-pcm16>"}
```

A WAV container is also accepted and is down-mixed to mono and resampled to 16 kHz.

Returns `{"embeddings": [[[...], ...]]}`: a batch of one embedding matrix, where each matrix holds one token/feature vector per row. The batch dimension is preserved.

### `POST /embed/docs`

Embed a batch of text documents.

```json
{"documents": ["First document text", "Second document text"]}
```

Returns `{"embeddings": [[[...], ...], ...]}`, one variable-length token embedding matrix per document.

### `POST /embed/query`

Embed one query. Provide exactly one of `text` or `audio`; `audio` uses the same base64 PCM16 (16 kHz mono) format as `/embed/audio`.

```json
{"text": "Find audio about carsickness"}
```

or

```json
{"audio": "<base64-pcm16>"}
```

Returns `{"embeddings": [[[...], ...]]}`: a batch of one embedding matrix, with the batch dimension preserved. Images are not accepted.

### `POST /v2/models/colqwen/infer` (MaxSim)

Scores precomputed query embeddings against document embeddings using ColBERT
late interaction. This is the KServe **V2 (Open Inference Protocol)** endpoint,
so inputs are named FP32 tensors. It returns one score matrix
`(num_queries, num_documents)`; entry `[i, j]` is `sum over query tokens of the
max dot product against document j's tokens`.

Batches are ragged (each query/document has its own token count), so embeddings
can be sent two ways:

- **Padded:** `queries`/`documents` shaped `(batch, tokens, dim)`, right-padded
  to a common length.
- **Flattened:** `queries`/`documents` shaped `(total_tokens, dim)` with a
  matching `query_lengths`/`document_lengths` INT tensor giving the per-item
  token counts.

`query_lengths`/`document_lengths` are optional for padded tensors (defaulting
to the full padded length) and required for flattened tensors. All tensors must
share the same embedding dimension.

```json
{
  "inputs": [
    {"name": "queries", "shape": [1, 2, 3], "datatype": "FP32",
     "data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
    {"name": "documents", "shape": [2, 2, 3], "datatype": "FP32",
     "data": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4]}
  ]
}
```

Returns:

```json
{
  "model_name": "colqwen",
  "id": "...",
  "outputs": [
    {"name": "scores", "shape": [1, 2], "datatype": "FP32", "data": [0.42, 0.84]}
  ]
}
```

## Configuration

The model is configured through environment variables (no code changes needed):

| Variable            | Default                                       | Description                                                                 |
| ------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| `HF_MODEL_NAME`     | `vidore/colqwen-omni-v0.1`                    | Audio-capable Hugging Face model id passed to `MultiVectorEncoder`.         |
| `SERVED_MODEL_NAME` | `colqwen`                                     | Model name registered with `ModelServer`; must match the deployment name.   |
| `DEVICE`            | `cpu`                                         | Torch device for the encoder, e.g. `cpu` or `cuda`.                          |
| `HF_TOKEN`          | *(unset)*                                     | Hugging Face token for gated/private repos.                                 |

The custom `/embed/...` routes are served by the same KServe HTTP server. KServe health endpoints remain available for probes.

## Docker

```bash
make build
# equivalent to:
# docker build -f docker/Dockerfile -t localhost/kserve_img:latest .
```

Run the image:

```bash
docker run -p 8080:8080 localhost/kserve_img:latest
```

> The current image installs dependencies and starts the server, but does **not** bake model weights into the image — the model is downloaded from Hugging Face on first start. See the next steps below.

## Deploy to serverless KServe (Helm)

`charts/colqwen-mve` renders a namespaced `ServingRuntime` (protocol `v2`) plus an `InferenceService` that references it. It targets KServe's serverless (Knative) mode.

```bash
# Point the chart at your pushed image first.
helm upgrade --install colqwen charts/colqwen-mve \
  --namespace kserve-test --create-namespace \
  --set image.repository=<registry>/<repo> \
  --set image.tag=<tag>
```

Common overrides (`charts/colqwen-mve/values.yaml`):

| Value                        | Purpose                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `model.name`                 | InferenceService + served model name (default `colqwen`).      |
| `model.hfModelName`          | Hugging Face repo id (`HF_MODEL_NAME`).                        |
| `model.storageUri`           | Optional KServe `storageUri` instead of downloading from HF.   |
| `runtime.kind`               | `ServingRuntime` or `ClusterServingRuntime`.                   |
| `runtime.device`             | Torch device (`DEVICE`); default `cpu` for CPU-only serving.   |
| `deployment.minReplicas`     | `1` keeps the model warm; `0` enables scale-to-zero.           |
| `resources.requests/limits` | CPU and memory for the predictor (CPU inference; memory-bound). |
| `predictor.nodeSelector` / `predictor.tolerations` | Schedule onto suitable nodes.               |
| `huggingface.tokenSecretName`| Secret holding an `HF_TOKEN` for gated models.                 |

After the `InferenceService` reports `READY=True`:

```bash
kubectl -n kserve-test get inferenceservice colqwen
```

`helm status colqwen -n kserve-test` prints the exact `curl` smoke test (it is also in `templates/NOTES.txt`).

## Next steps for production deployment

The repository currently targets local development. The following work is required before production use.

### 1. Dependency and model reproducibility

- [ ] Pin versions in `requirements.txt` (or adopt a lock file such as `uv.lock` / `pip-compile`). The `sentence_transformers.MultiVectorEncoder` API requires a specific `sentence-transformers` release; unpinned installs are not reproducible.
- [ ] Pin/record the exact model revision (`vidore/colqwen-omni-v0.1`) instead of tracking the default branch.
- [ ] Bake weights into the image or mount a pre-populated model cache volume to eliminate cold-start downloads and external network dependency.

### 2. Container hardening

- [ ] Switch to a CUDA-enabled base image (e.g. `nvidia/cuda` + matching PyTorch) if you later move inference to GPU.
- [ ] Run the container as a non-root user.
- [ ] Add a `.dockerignore` to keep the build context small and avoid leaking local files.
- [ ] Add a `HEALTHCHECK` hitting the model readiness endpoint.
- [ ] Consider a multi-stage build to reduce final image size.

### 3. Serving and scaling

- [ ] Keep the KServe runtime protocol and custom `/embed/...` routes aligned with the deployed ingress configuration.
- [ ] Configure node scheduling, resource requests/limits, and `minReplicas` for warm capacity.
- [ ] Enable autoscaling and request batching (CPU inference is the throughput bottleneck).
- [ ] Define readiness/liveness probes against the KServe ready endpoint.

### 4. Configuration

- [ ] Make the 16 kHz audio sample rate configurable if deployments require a different model processor rate.
- [ ] Add explicit request size limits for text and base64 audio payloads.

### 5. Observability

- [ ] Structured JSON logging with request IDs.
- [ ] Metrics for request latency, throughput, batch size, and CPU utilization.
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

- [ ] Unit tests for request parsing, WAV decoding/resampling, and embedding serialization.
- [ ] Integration tests against a running server for all three `/embed/...` endpoints.
- [ ] Load tests to validate latency under batch/concurrency.

### 9. Input robustness

- [ ] Validate and cap WAV payload sizes and text lengths to protect memory.
