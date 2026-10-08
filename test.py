"""Smoke test for the ColQwen-Omni embedding service.

Uses only the standard library. Exercises every endpoint and prints the shape
of each returned embedding, e.g. ``[[14, 128], [18, 128]]`` for a two-document
batch (tokens x dim, variable per input).

Usage:
    python test.py [--base-url http://localhost:8080] [--skip-audio] [--skip-maxsim]
"""

import argparse
import base64
import json
import math
import struct
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8080"
# Must match SERVED_MODEL_NAME on the server.
MODEL_NAME = "colqwen"
# The audio format contract: little-endian PCM16 mono at 16 kHz.
AUDIO_RATE = 16_000


def post(base_url: str, path: str, body: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        print(f"FAIL {path}: HTTP {error.code}: {error.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as error:
        print(f"FAIL {path}: {error.reason}", file=sys.stderr)
        sys.exit(1)


def matrix_shape(matrix: list) -> list:
    """Shape of one (tokens, dim) embedding matrix as [tokens, dim]."""
    return [len(matrix), len(matrix[0]) if matrix else 0]


def print_shapes(label: str, embeddings: list) -> None:
    print(f"{label:<12} shapes: {json.dumps([matrix_shape(m) for m in embeddings])}")


def pcm16_sine(seconds: float = 0.5, frequency: float = 440.0) -> bytes:
    """Synthesize headerless little-endian PCM16 mono samples at 16 kHz."""
    count = int(seconds * AUDIO_RATE)
    return struct.pack(
        f"<{count}h",
        *(int(12000 * math.sin(2 * math.pi * frequency * i / AUDIO_RATE)) for i in range(count)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument("--skip-audio", action="store_true", help="skip /embed/audio")
    parser.add_argument("--skip-maxsim", action="store_true", help="skip the infer round-trip")
    args = parser.parse_args()
    base_url = args.base_url

    documents = post(
        base_url,
        "/embed/docs",
        {
            "documents": [
                "A recording describes a person seeking a taxi.",
                "A short conversation about finding a book in a shop.",
            ]
        },
    )["embeddings"]
    print_shapes("embed/docs", documents)

    query = post(
        base_url,
        "/embed/query",
        {"text": "A person looking for a taxi"},
    )["embeddings"]
    print_shapes("embed/query", query)

    if not args.skip_audio:
        audio = post(
            base_url,
            "/embed/audio",
            {"audio": base64.b64encode(pcm16_sine()).decode()},
        )["embeddings"]
        print_shapes("embed/audio", audio)

    if not args.skip_maxsim:
        # Flatten the ragged batches and score query against the two documents.
        flat_query = [vector for matrix in query for vector in matrix]
        flat_docs = [vector for matrix in documents for vector in matrix]
        response = post(
            base_url,
            f"/v2/models/{MODEL_NAME}/infer",
            {
                "inputs": [
                    {
                        "name": "queries",
                        "shape": matrix_shape(flat_query),
                        "datatype": "FP32",
                        "data": [x for vector in flat_query for x in vector],
                    },
                    {
                        "name": "query_lengths",
                        "shape": [len(query)],
                        "datatype": "INT32",
                        "data": [len(matrix) for matrix in query],
                    },
                    {
                        "name": "documents",
                        "shape": matrix_shape(flat_docs),
                        "datatype": "FP32",
                        "data": [x for vector in flat_docs for x in vector],
                    },
                    {
                        "name": "document_lengths",
                        "shape": [len(documents)],
                        "datatype": "INT32",
                        "data": [len(matrix) for matrix in documents],
                    },
                ]
            },
        )
        output = response["outputs"][0]
        scores = output["data"]
        num_docs = len(documents)
        rows = [scores[i * num_docs : (i + 1) * num_docs] for i in range(len(query))]
        print(f"maxsim       shape: {output['shape']}")
        print(f"maxsim       scores: {rows}")

    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
