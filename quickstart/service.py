from __future__ import annotations
from bge3.bento_service import BGEService
import grpc
from concurrent import futures
from quickstart.generated.model_service_pb2_grpc import add_SimilarityServiceServicer_to_server


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SimilarityServiceServicer_to_server(BGEService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC Server is running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

