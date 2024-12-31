from __future__ import annotations
import bentoml
from FlagEmbedding import FlagAutoModel
from transformers import AutoTokenizer

import grpc
from concurrent import futures
from model_service_pb2 import SimilarityRequest, SimilarityResponse
from model_service_pb2_grpc import SimilarityServiceServicer, add_SimilarityServiceServicer_to_server

@bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
class MyService(SimilarityServiceServicer):
    def __init__(self) -> None:
        self.model = FlagAutoModel.from_finetuned(
            'BAAI/bge-base-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

    def ComputeSimilarity(self, request: SimilarityRequest,
                          context) -> SimilarityResponse:
        embeddings_1 = self.model.encode(request.sentences_1)
        embeddings_2 = self.model.encode(request.sentences_2)

        similarity = embeddings_1 @ embeddings_2.T

        print("Raw similarity result:", similarity)

        similarity_values = similarity.tolist()

        print("Similarity values:", similarity_values)

        response = SimilarityResponse(similarity=similarity_values)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SimilarityServiceServicer_to_server(MyService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC Server is running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()