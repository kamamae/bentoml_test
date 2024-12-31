import grpc
from model_service_pb2 import SimilarityRequest
from model_service_pb2_grpc import SimilarityServiceStub


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = SimilarityServiceStub(channel)

    sentences_1 = ["Sentence one", "Sentence two"]
    sentences_2 = ["Another sentence", "Yet another sentence"]

    request = SimilarityRequest(sentences_1=sentences_1, sentences_2=sentences_2)
    response = stub.ComputeSimilarity(request)

    print("Similarity:", response.similarity)


if __name__ == "__main__":
    run()