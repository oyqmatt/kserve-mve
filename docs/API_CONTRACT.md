# ColQwen Omni Embedding API — Contract v1

This document is the stable contract for clients of the ColQwen-Omni embedding
service. Anything not specified here may change; anything specified here follows
[semantic versioning](#versioning): breaking changes require a major version
bump and will be documented.

---

## 1. Conventions

| Aspect | Contract |
| --- | --- |
| Transport | HTTP/1.1, JSON bodies, `Content-Type: application/json` |
| Base URL | Deployment-specific (port `8080` locally; KServe/Knative host otherwise) |
| Encoding | All text is UTF-8; all audio is little-endian |
| Embedding model | `vidore/colqwen-omni-v0.1` by default; dimension is model-dependent (**128** for the default) |
| Multi-vector semantics | Each input yields one vector per token/feature: a matrix of shape `(tokens, dim)` |
| Ordering | The *i*-th returned embedding always corresponds to the *i*-th input |
| Authentication | None enforced at the service; may exist at your ingress (TLS, auth) |

### 1.1 What "multi-vector" means for clients

- An embedding is a **list of matrices**, not a single vector.
- Token counts are **variable per input** — never assume a fixed first dimension.
  A short text may produce `~12 × 128`; a document page `~1000 × 128`.
- Responses are **neither flattened nor padded**: the JSON nesting *is* the shape
  (`embeddings[i]` has exactly `tokens_i` rows of `dim` floats).
- Requests containing a single item still return a **1-element batch**.

### 1.2 Performance envelope (CPU deployment)

The default deployment runs inference on CPU:

- Expect **seconds per request**, scaling with input length and batch size.
- Requests are processed through a small worker pool; concurrent requests queue
  rather than batch.
- Keep document batches modest (tens, not thousands) and reuse embeddings:
  embed a document **once**, store it, and score with the MaxSim endpoint later.

---

## 2. Embedding endpoints

All three endpoints share one response shape:

```json
{"embeddings": [[[ ...token vectors... ]]]}
```

| Level | Contents |
| --- | --- |
| `embeddings` | Batch: one entry per input, in input order |
| `embeddings[i]` | One matrix: `tokens_i` rows, each of `dim` floats |
| `embeddings[i][k]` | One token/feature vector: `dim` JSON numbers (float32 values) |

### 2.1 `POST /embed/audio`

Embeds one audio clip as a **document**.

**Request body**

| Field | Type | Constraints |
| --- | --- | --- |
| `audio` | string | Required. Base64-encoded audio (plain or `data:` URI containing `;base64`) |

**Accepted audio formats** (in priority order):

1. **Raw PCM16 (primary contract):** already-decoded, headerless,
   **little-endian signed 16-bit, mono, 16 kHz** samples. The service converts
   them to float32 in `[-1, 1]` without resampling. Byte length must be even.
2. **WAV container (compatibility):** a RIFF/WAV file. It is down-mixed to mono
   and resampled to 16 kHz. Non-WAV containers (FLAC, MP3, Ogg, MP4, …) are
   rejected — decode to PCM16 or WAV on the client first.

Clients that already control their audio pipeline **should send raw PCM16**:
it avoids container parsing, sample-rate guessing, and resampling differences
between senders.

**Example**

```bash
curl -s http://localhost:8080/embed/audio \
  -H 'Content-Type: application/json' \
  -d "{\"audio\": \"$(base64 -w0 clip.pcm16)\"}"
```

**Response**: the shared shape above, with exactly **one** entry in the batch.

### 2.2 `POST /embed/docs`

Embeds a batch of **text documents**.

**Request body**

| Field | Type | Constraints |
| --- | --- | --- |
| `documents` | string[] | Required. Non-empty array; each entry a non-empty, non-whitespace string |

**Example**

```bash
curl -s http://localhost:8080/embed/docs \
  -H 'Content-Type: application/json' \
  -d '{"documents": ["First document text", "Second document text"]}'
```

**Response**: the shared shape above, one matrix per document.

### 2.3 `POST /embed/query`

Embeds **one query**, either text or audio. This mirrors the asymmetric
query/document encoding of ColBERT-style models: a query must be encoded with
the query path to match document embeddings scored by the MaxSim endpoint.

**Request body** — provide **exactly one** field:

| Field | Type | Constraints |
| --- | --- | --- |
| `text` | string | Non-empty, non-whitespace |
| `audio` | string | Same formats and constraints as `/embed/audio` |

Sending both — or neither — is an error.

**Examples**

```bash
curl -s http://localhost:8080/embed/query \
  -H 'Content-Type: application/json' \
  -d '{"text": "A person looking for a taxi"}'
```

**Response**: the shared shape above, with exactly **one** entry.

---

## 3. `POST /v2/models/{model}/infer` — MaxSim scoring

Computes ColBERT **late interaction** scores from embeddings previously produced
by the embedding endpoints. `{model}` is the served model name (`colqwen` by
default; it must match `SERVED_MODEL_NAME`).

This is the [KServe V2 / Open Inference Protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol)
endpoint. Inputs are named tensors in the V2 envelope.

**Semantics.** With query embedding matrix $Q \in \mathbb{R}^{q \times d}$ and
document matrix $D \in \mathbb{R}^{t \times d}$, the score for one pair is:

$$
\operatorname{score}(Q, D) \;=\; \sum_{i=1}^{q} \max_{j=1..t} \; Q_i \cdot D_j
$$

i.e. for each query token, take the maximum dot product against any document
token, then sum over query tokens. Higher is more relevant. Scores are
**unnormalized dot products** — their scale depends on the model; do not compare
them across different models.

**Inputs**

| Tensor | Datatype | Shape | Required |
| --- | --- | --- | --- |
| `queries` | FP32 | `(total_tokens, dim)` **or** `(batch, tokens, dim)` | yes |
| `query_lengths` | INT32/INT64 | `(batch,)` | required if `queries` is 2D |
| `documents` | FP32 | `(total_tokens, dim)` **or** `(batch, tokens, dim)` | yes |
| `document_lengths` | INT32/INT64 | `(batch,)` | required if `documents` is 2D |

Ragged batches (different token counts per item) can be sent two ways:

- **Flattened (recommended):** all item matrices concatenated along the token
  axis as `(total_tokens, dim)`, plus a lengths tensor of per-item token counts.
  The lengths must sum exactly to `total_tokens`.
- **Padded:** `(batch, tokens, dim)` right-padded to a common length, plus
  optional lengths (default: full length; entries larger than the padded token
  count are rejected).

All four tensors must share the same embedding dimension.

**Example request** (padded, 1 query × 2 documents, dim 3):

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

**Response**

| Output | Datatype | Shape | Contents |
| --- | --- | --- | --- |
| `scores` | FP32 | `(num_queries, num_documents)` | Row-major **flattened** in `data` |

```json
{
  "model_name": "colqwen",
  "id": "…",
  "outputs": [
    {"name": "scores", "shape": [1, 2], "datatype": "FP32", "data": [12.34, 9.87]}
  ]
}
```

`data` is the score matrix flattened **row-major**: for shape `[Q, D]`,
`data[i*D + j]` is the score of query `i` against document `j`. The response
`id` echoes the request `id` when provided, for correlating concurrent requests.

---

## 4. Errors

Two error styles exist, matching the two route families:

### `/embed/*` (FastAPI-style)

```json
{"detail": "'documents' must be a non-empty array"}
```

| Status | When |
| --- | --- |
| `422` | Missing/invalid fields, invalid base64, non-WAV audio, odd-length PCM16, both/neither of `text`+`audio`, empty documents |
| `503` | Model still loading (`"Encoder is not ready"`) — retry with backoff |

### `/v2/...` (KServe-style)

```json
{"error": "…"}
```

| Status | When |
| --- | --- |
| `400` | Missing/invalid tensors: wrong datatype, ragged 2D without lengths, lengths sum mismatch, dimension mismatch |
| `404` | Unknown `{model}` name in the URL |
| `503` | Model still loading |

### Client guidance

- Treat `4xx` as **do not retry** (fix the payload); `503`/`5xx` as **retryable**
  with exponential backoff.
- Under Knative scale-to-zero, the first request may be queued while the pod
  starts; use generous timeouts (tens of seconds) or `minReplicas: 1`.

---

## 5. Operational endpoints

| Method & path | Purpose |
| --- | --- |
| `GET /v2/health/live` | Process is up (liveness probe) |
| `GET /v2/health/ready` | All registered models are loaded (readiness probe) |
| `GET /v2/models/{model}/ready` | Specific model readiness |
| `GET /metrics` | Prometheus metrics (KServe default) |

An interactive API schema is served at `/docs` only when the server is started
with `--enable_docs_url` (disabled by default in KServe).

---

## 6. Guarantees, limits, and known gaps

**Guaranteed (v1):**

- Input-order correspondence between request and `embeddings`.
- No flattening, no padding — shapes in §1.1.
- Raw PCM16 accepted as specified; WAV accepted as compatibility.
- Scores computed exactly as §3 defines, in float32.

**Not yet guaranteed (roadmap):**

- **No payload size caps** are enforced server-side yet. Until then, clients
  must self-limit: keep audio clips to a few minutes of 16 kHz PCM16 (~128 KiB
  per second of audio → a 5-minute clip is ~38 MB, ~50 MB base64) and text
  documents to a few thousand characters.
- **No authentication, rate limiting, or request quotas** at the service level.
- **No top-k retrieval** — the infer endpoint scores every query/document pair
  sent; filter the document set on the client for large corpora.
- **No persistence** — the service is stateless; clients store embeddings.

---

## 7. Minimal client recipe (Python)

```python
import base64
import json
import urllib.request

BASE = "http://localhost:8080"

def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)

# 1. Embed documents once, store the result.
docs = post("/embed/docs", {"documents": ["doc one", "doc two"]})["embeddings"]

# 2. Embed a query.
query = post("/embed/query", {"text": "about carsickness"})["embeddings"]

# 3. Score the query against the stored documents via MaxSim.
flat_q = [vec for mat in query for vec in mat]          # (total_q, 128)
flat_d = [vec for mat in docs for vec in mat]           # (total_d, 128)
resp = post(
    f"/v2/models/colqwen/infer",
    {
        "inputs": [
            {
                "name": "queries",
                "shape": [len(flat_q), len(flat_q[0])],
                "datatype": "FP32",
                "data": [x for vec in flat_q for x in vec],
            },
            {
                "name": "query_lengths",
                "shape": [len(query)],
                "datatype": "INT32",
                "data": [len(mat) for mat in query],
            },
            {
                "name": "documents",
                "shape": [len(flat_d), len(flat_d[0])],
                "datatype": "FP32",
                "data": [x for vec in flat_d for x in vec],
            },
            {
                "name": "document_lengths",
                "shape": [len(docs)],
                "datatype": "INT32",
                "data": [len(mat) for mat in docs],
            },
        ]
    },
)
scores = resp["outputs"][0]["data"]  # row-major: [q0_d0, q0_d1, …]
best = max(range(len(docs)), key=lambda j: scores[j])
```

---

## 8. Versioning

- Contract version: **1** (this document).
- Additive changes (new optional request fields, new endpoints, new response
  fields) are minor-version bumps; existing fields keep meaning and position.
- Breaking changes (removing/renaming fields, changing shapes or formats)
  require a major bump and a migration note in this document.
- The response header middleware adds a trace ID; per-request IDs are echoed in
  the infer response `id`.
