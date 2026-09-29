import argparse
import os
import time
from typing import Dict, List, Union

import kserve
from kserve import InferOutput, InferRequest, InferResponse, Model, ModelServer
from kserve.errors import InvalidInput
from kserve.utils.utils import generate_uuid

from sentence_transformers import MultiVectorEncoder

DEFAULT_MODEL_NAME = "mxbai-edge-colbert-v0-17m"
DEFAULT_MODEL_PATH = "mixedbread-ai/mxbai-edge-colbert-v0-17m"


class MVE(Model):
    def __init__(self, name: str, model_path: str = DEFAULT_MODEL_PATH):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.model_path = model_path
        self.load()

    def load(self):
        self.model = MultiVectorEncoder(self.model_path)
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def get_input_types(self) -> List[Dict]:
        return [
            {"name": "queries", "datatype": "BYTES", "shape": [-1]},
            {"name": "documents", "datatype": "BYTES", "shape": [-1]},
        ]

    def get_output_types(self) -> List[Dict]:
        return [{"name": "predictions", "datatype": "FP32", "shape": [-1, -1]}]

    @staticmethod
    def _decode_bytes_input(request: InferRequest, name: str) -> List[str]:
        """Extract a flat list of UTF-8 strings from a v2 BYTES input tensor."""
        infer_input = request.get_input_by_name(name)
        if infer_input is None:
            raise InvalidInput(f"Missing required input tensor '{name}'")
        if infer_input.datatype != "BYTES":
            raise InvalidInput(
                f"Input tensor '{name}' must have datatype BYTES, got {infer_input.datatype}"
            )

        data = infer_input.data
        if data is None or (not isinstance(data, (str, bytes)) and len(data) == 0):
            data = infer_input.as_numpy().flatten().tolist()
        elif isinstance(data, (str, bytes)):
            data = [data]

        flattened: List = []
        for item in data:
            if isinstance(item, (list, tuple)):
                flattened.extend(item)
            else:
                flattened.append(item)

        return [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in flattened
        ]

    async def predict(
        self,
        payload: Union[Dict, InferRequest],
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Union[Dict, InferResponse]:
        start = time.time()

        # v2 endpoints pass an InferRequest, v1 endpoints pass a plain dict.
        if isinstance(payload, InferRequest):
            queries = self._decode_bytes_input(payload, "queries")
            documents = self._decode_bytes_input(payload, "documents")
        else:
            queries = payload.get("queries", [])
            documents = payload.get("documents", [])

        query_embeddings = self.model.encode_query(queries)
        document_embeddings = self.model.encode_document(documents)
        scores = self.model.similarity(query_embeddings, document_embeddings).tolist()
        print(f"Prediction time: {time.time() - start:.2f} seconds")

        if isinstance(payload, InferRequest):
            num_queries = len(scores)
            num_documents = len(scores[0]) if scores else 0
            return InferResponse(
                response_id=payload.id or generate_uuid(),
                model_name=self.name,
                infer_outputs=[
                    InferOutput(
                        name="predictions",
                        shape=[num_queries, num_documents],
                        datatype="FP32",
                        data=[float(score) for row in scores for score in row],
                    )
                ],
            )

        return {"predictions": scores}


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # Configure kserve and uvicorn logger
    model = MVE(
        os.environ.get("SERVED_MODEL_NAME", DEFAULT_MODEL_NAME),
        os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL_PATH),
    )
    ModelServer().start([model])
