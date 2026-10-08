"""KServe service exposing ColQwen-Omni multi-vector embedding and MaxSim scoring.

Each input is encoded into one vector per token (or audio feature) instead of a
single pooled vector, ColBERT-style. The ``/embed/...`` endpoints return those
embeddings; the KServe V2 ``infer`` endpoint scores precomputed query and
document embeddings with ColBERT MaxSim late interaction.
"""

import asyncio
import base64
import binascii
import io
import math
import os

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from kserve import (
    InferInput,
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
)
from kserve.errors import InvalidInput
from kserve.model_server import app
from kserve.utils.utils import generate_uuid
from scipy.signal import resample_poly
from sentence_transformers import MultiVectorEncoder

DEFAULT_MODEL_NAME = "colqwen"
DEFAULT_MODEL_PATH = "vidore/colqwen-omni-v0.1"

# The audio tower expects mono float32 at 16 kHz.
AUDIO_SAMPLE_RATE = 16_000
PCM16_DTYPE = np.dtype("<i2")
PCM16_SCALE = 32768.0
WAV_MAGIC = b"WAVE"

# ColQwen-Omni does not fit a consumer GPU, so it runs on CPU by default.
DEFAULT_DEVICE = "cpu"

# V2 tensor names for the MaxSim infer endpoint.
QUERY_INPUT = "queries"
QUERY_LENGTHS_INPUT = "query_lengths"
DOCUMENT_INPUT = "documents"
DOCUMENT_LENGTHS_INPUT = "document_lengths"
SCORES_OUTPUT = "scores"


class ColQwenModel(Model):
    """KServe model wrapper that owns the encoder and its readiness flag."""

    def __init__(
        self,
        name: str,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = DEFAULT_DEVICE,
    ):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.model_path = model_path
        self.device = device
        self.load()

    def load(self):
        encoder = MultiVectorEncoder(self.model_path, device=self.device)
        if not encoder.supports("audio"):
            raise ValueError(
                f"Model '{self.model_path}' must support audio for this service"
            )
        app.state.encoder = encoder
        # The ready flag gates the model ready endpoint used by readiness probes.
        self.ready = True

    def get_input_types(self) -> list[dict]:
        return [
            {"name": QUERY_INPUT, "datatype": "FP32", "shape": [-1, -1, -1]},
            {"name": QUERY_LENGTHS_INPUT, "datatype": "INT32", "shape": [-1]},
            {"name": DOCUMENT_INPUT, "datatype": "FP32", "shape": [-1, -1, -1]},
            {"name": DOCUMENT_LENGTHS_INPUT, "datatype": "INT32", "shape": [-1]},
        ]

    def get_output_types(self) -> list[dict]:
        return [{"name": SCORES_OUTPUT, "datatype": "FP32", "shape": [-1, -1]}]

    async def predict(
        self,
        payload: InferRequest,
        headers: dict[str, str] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> InferResponse:
        """Score query embeddings against document embeddings with MaxSim.

        Expects precomputed token embeddings from the ``/embed/...`` endpoints.
        ``queries``/``documents`` are FP32 tensors that may be ragged batches,
        shaped ``(batch, tokens, dim)`` when right-padded or ``(total_tokens,
        dim)`` when flattened. Flattened inputs require a matching
        ``query_lengths``/``document_lengths`` INT tensor with the per-item
        token counts. Returns an FP32 ``(num_queries, num_documents)`` matrix.
        """
        if not isinstance(payload, InferRequest):
            raise InvalidInput(
                "MaxSim scoring requires a KServe V2 InferRequest; "
                "POST to /v2/models/<name>/infer"
            )

        queries = _decode_embedding_batch(
            payload, QUERY_INPUT, QUERY_LENGTHS_INPUT
        )
        documents = _decode_embedding_batch(
            payload, DOCUMENT_INPUT, DOCUMENT_LENGTHS_INPUT
        )
        scores = await asyncio.to_thread(_maxsim, queries, documents)

        return InferResponse(
            response_id=payload.id or generate_uuid(),
            model_name=self.name,
            infer_outputs=[
                InferOutput(
                    name=SCORES_OUTPUT,
                    shape=list(scores.shape),
                    datatype="FP32",
                    data=scores.flatten().tolist(),
                )
            ],
        )


def _encoder() -> MultiVectorEncoder:
    """Return the loaded encoder, failing with 503 until the model is ready."""
    encoder = getattr(app.state, "encoder", None)
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder is not ready")
    return encoder


def _tensor(request: InferRequest, name: str) -> InferInput:
    """Return a named input tensor, raising InvalidInput when it is absent."""
    infer_input = request.get_input_by_name(name)
    if infer_input is None:
        raise InvalidInput(f"Missing required input tensor '{name}'")
    return infer_input


def _decode_embedding_batch(
    request: InferRequest, data_name: str, lengths_name: str
) -> list[np.ndarray]:
    """Split a batched token-embedding tensor into one (tokens, dim) matrix each.

    Accepts either a right-padded ``(batch, tokens, dim)`` tensor, or a flat
    ``(total_tokens, dim)`` tensor paired with a ``lengths_name`` tensor holding
    the per-item token counts.
    """
    infer_input = _tensor(request, data_name)
    if infer_input.datatype != "FP32":
        raise InvalidInput(
            f"Input tensor '{data_name}' must have datatype FP32, "
            f"got {infer_input.datatype}"
        )

    embeddings = infer_input.as_numpy()
    if embeddings.ndim not in (2, 3):
        raise InvalidInput(
            f"Input tensor '{data_name}' must be 2D (total_tokens, dim) or "
            f"3D (batch, tokens, dim), got shape {list(embeddings.shape)}"
        )

    lengths_input = request.get_input_by_name(lengths_name)
    if embeddings.ndim == 2:
        if lengths_input is None:
            raise InvalidInput(
                f"'{lengths_name}' is required when '{data_name}' is flattened "
                "(2D)"
            )
        lengths = _decode_lengths(lengths_input, lengths_name)
        total = int(lengths.sum())
        if total != embeddings.shape[0]:
            raise InvalidInput(
                f"'{lengths_name}' sums to {total} but '{data_name}' has "
                f"{embeddings.shape[0]} rows"
            )
        return list(np.split(embeddings, np.cumsum(lengths)[:-1]))

    batch, tokens, _ = embeddings.shape
    if lengths_input is None:
        lengths = np.full(batch, tokens, dtype=np.int64)
    else:
        lengths = _decode_lengths(lengths_input, lengths_name)
        if lengths.size != batch:
            raise InvalidInput(
                f"'{lengths_name}' must have one entry per batch item "
                f"({batch}), got {lengths.size}"
            )
        if np.any(lengths > tokens):
            raise InvalidInput(
                f"'{lengths_name}' exceeds the padded token count ({tokens})"
            )
    return [embeddings[i, : lengths[i], :] for i in range(batch)]


def _decode_lengths(infer_input: InferInput, name: str) -> np.ndarray:
    """Read a 1D tensor of positive per-item token counts."""
    lengths = infer_input.as_numpy().astype(np.int64).reshape(-1)
    if lengths.size == 0 or np.any(lengths <= 0):
        raise InvalidInput(f"'{name}' must contain positive token counts")
    return lengths


def _maxsim(queries: list[np.ndarray], documents: list[np.ndarray]) -> np.ndarray:
    """Compute ColBERT late-interaction scores for all query/document pairs."""
    if not queries:
        raise InvalidInput(f"'{QUERY_INPUT}' must contain at least one item")
    if not documents:
        raise InvalidInput(f"'{DOCUMENT_INPUT}' must contain at least one item")

    dim = queries[0].shape[1]
    if any(query.shape[1] != dim for query in queries):
        raise InvalidInput("All query embeddings must share the same dimension")
    if any(document.shape[1] != dim for document in documents):
        raise InvalidInput(
            "Query and document embeddings must share the same dimension"
        )

    scores = np.empty((len(queries), len(documents)), dtype=np.float32)
    for j, document in enumerate(documents):
        # Inputs are guaranteed FP32 by _decode_embedding_batch; transposing
        # once per document avoids recomputing it for every query.
        document_t = document.T
        for i, query in enumerate(queries):
            similarity = query @ document_t
            scores[i, j] = similarity.max(axis=1).sum()
    return scores


def _decode_base64(value: object, field: str) -> bytes:
    """Decode a plain or data-URI base64 string into bytes."""
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=422, detail=f"'{field}' must be a base64 encoded string"
        )

    encoded = value
    if encoded.startswith("data:"):
        header, _, encoded = encoded.partition(",")
        if not encoded or ";base64" not in header:
            raise HTTPException(
                status_code=422, detail=f"'{field}' must be a base64 data URI"
            )

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"'{field}' is not valid base64"
        ) from exc


def _decode_pcm16(data: bytes, field: str) -> np.ndarray:
    """Interpret headerless little-endian 16-bit samples as a float32 waveform."""
    if not data or len(data) % PCM16_DTYPE.itemsize:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{field}' must be non-empty little-endian PCM16 "
                "(16-bit) mono audio"
            ),
        )
    samples = np.frombuffer(data, dtype=PCM16_DTYPE)
    return samples.astype(np.float32) / PCM16_SCALE


def _decode_wav(data: bytes, field: str) -> np.ndarray:
    """Decode a WAV container into a mono 16 kHz float32 waveform."""
    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(data), dtype="float32", always_2d=True
        )
    except (sf.LibsndfileError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"'{field}' is not a readable WAV"
        ) from exc

    if sample_rate <= 0 or waveform.size == 0:
        raise HTTPException(
            status_code=422,
            detail=f"'{field}' WAV is empty or has an invalid sample rate",
        )

    mono = waveform.mean(axis=1)
    if sample_rate != AUDIO_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), AUDIO_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            AUDIO_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )
    return np.ascontiguousarray(mono, dtype=np.float32)


def _decode_audio(value: object, field: str = "audio") -> dict:
    """Build a model audio input from raw PCM16 or, if detected, a WAV container."""
    data = _decode_base64(value, field)
    is_wav = len(data) >= 12 and data[8:12] == WAV_MAGIC
    waveform = _decode_wav(data, field) if is_wav else _decode_pcm16(data, field)
    return {"audio": {"array": waveform, "sampling_rate": AUDIO_SAMPLE_RATE}}


def _embeddings_as_lists(embeddings: list) -> list[list[list[float]]]:
    """Convert torch/numpy token embeddings into JSON-serializable nested lists."""
    converted = []
    for embedding in embeddings:
        if hasattr(embedding, "detach"):  # torch.Tensor, possibly bfloat16
            embedding = embedding.detach().float().cpu()
        if hasattr(embedding, "tolist"):  # torch.Tensor or numpy array
            embedding = embedding.tolist()
        converted.append(embedding)
    return converted


def _require_text(value: object, field: str) -> str:
    """Validate that a payload field is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=422, detail=f"'{field}' must be a non-empty string"
        )
    return value


def _require_texts(value: object, field: str) -> list[str]:
    """Validate that a payload field is a non-empty list of non-empty strings."""
    if not isinstance(value, list) or not value:
        raise HTTPException(
            status_code=422, detail=f"'{field}' must be a non-empty array"
        )
    return [_require_text(item, f"{field}[]") for item in value]


async def _encode(encode_fn, inputs: list) -> dict:
    """Run a CPU-bound encode off the event loop so health probes stay responsive."""
    embeddings = await asyncio.to_thread(encode_fn, inputs)
    return {"embeddings": _embeddings_as_lists(embeddings)}


@app.post("/embed/audio")
async def embed_audio(payload: dict) -> dict:
    """Embed a single audio document supplied as base64 PCM16 (or WAV)."""
    audio = _decode_audio(payload.get("audio"))
    return await _encode(_encoder().encode_document, [audio])


@app.post("/embed/docs")
async def embed_docs(payload: dict) -> dict:
    """Embed a batch of text documents."""
    documents = _require_texts(payload.get("documents"), "documents")
    return await _encode(_encoder().encode_document, documents)


@app.post("/embed/query")
async def embed_query(payload: dict) -> dict:
    """Embed a single query, either text or audio (mutually exclusive)."""
    has_text = "text" in payload
    if has_text == ("audio" in payload):
        raise HTTPException(
            status_code=422, detail="Provide exactly one of 'text' or 'audio'"
        )

    query = (
        _require_text(payload["text"], "text")
        if has_text
        else _decode_audio(payload["audio"])
    )
    return await _encode(_encoder().encode_query, [query])


def main() -> None:
    # Configuration is env-driven (see README); KServe parses its own CLI args
    # at import, so no local parser is needed.
    model = ColQwenModel(
        os.environ.get("SERVED_MODEL_NAME", DEFAULT_MODEL_NAME),
        os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL_PATH),
        os.environ.get("DEVICE", DEFAULT_DEVICE),
    )
    ModelServer().start([model])


if __name__ == "__main__":
    main()
