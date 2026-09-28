import argparse
import time

from cffi import model
import torch
from PIL import Image
from typing import Dict

import kserve
from kserve import Model, ModelServer, logging
from kserve.model_server import app
from kserve.utils.utils import generate_uuid

from sentence_transformers import MultiVectorEncoder

class MVE(Model):
    def __init__(self, name: str, model_path = "mixedbread-ai/mxbai-edge-colbert-v0-17m"):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.model_path = model_path
        self.load()

    def load(self):
        self.model = MultiVectorEncoder(self.model_path)
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    async def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Dict:
        start = time.time()
        queries = payload.get("queries", [])
        documents = payload.get("documents", [])

        # queries = [
        #     "What is the variable represented on the y-axis of the graph?",
        #     "Total outlay is maximum in which year?",
        # ]
        # documents = [
        #     "Venus is often called Earth's twin because of its similar size and proximity.",
        #     "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        #     "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        #     "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        # ]

        query_embeddings = self.model.encode_query(queries)
        document_embeddings = self.model.encode_document(documents)
        scores = self.model.similarity(query_embeddings, document_embeddings)
        print(f"Prediction time: {time.time() - start:.2f} seconds")
        # Convert tensors to lists for JSON serialization
        scores = scores.tolist()
        return {"predictions": scores}


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # Configure kserve and uvicorn logger
    model = MVE("mxbai")
    ModelServer().start([model])