from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen-omni-v0.1")

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
documents = [
    f"https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc{i}.jpg" for i in range(1, 5)
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(f"Query 0 shape:    {tuple(query_embeddings[0].shape)}")
print(f"Document 0 shape: {tuple(document_embeddings[0].shape)}")
# Query 0 shape:    (61, 128)
# Document 0 shape: (1034, 128)

# MaxSim late-interaction scoring (rows = queries, columns = documents)
scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[53.5625, 49.2036, 46.6958, 45.4949],
#         [45.6436, 53.1328, 45.0957, 45.5176]])
