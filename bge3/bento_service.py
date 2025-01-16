from __future__ import annotations
import bentoml
from quickstart.generated.model_service_pb2 import SimilarityRequest, SimilarityResponse
from quickstart.generated.model_service_pb2_grpc import SimilarityServiceServicer
from FlagEmbedding import BGEM3FlagModel

@bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
class BGEService(SimilarityServiceServicer):
    def __init__(self) -> None:
        self.model = BGEM3FlagModel('BAAI/bge-m3',
                               use_fp16=True)

    def ComputeSimilarity(self, request: SimilarityRequest,
                          context) -> SimilarityResponse:
        embeddings_1 = self.model.encode(request.sentences_1,
                                    batch_size=12,
                                    max_length=8192,
                                    )['dense_vecs']
        embeddings_2 = self.model.encode(request.sentences_2)['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_values = similarity.flatten().tolist()  # Для одномерного списка
        response = SimilarityResponse(similarity=similarity_values)
        print("Similarity values:", similarity_values)

        return response

    def SparseEmbedding(self, request: SimilarityRequest,
                        context) -> SimilarityResponse:
        embeddings_1 = list(request.sentences_1)
        embeddings_2 = list(request.sentences_2)

        print(f"Received sentences_1: {embeddings_1}, type: {type(embeddings_1)}")
        print(f"Received sentences_2: {embeddings_2}, type: {type(embeddings_2)}")


        output_1 = self.model.encode(embeddings_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        output_2 = self.model.encode(embeddings_2, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        print(self.model.convert_id_to_token(output_1['lexical_weights']))

        return self.model.convert_id_to_token(output_1['lexical_weights'])

    def ColBERT(self, request: SimilarityRequest, context) -> SimilarityResponse:
        embeddings_1 = list(request.sentences_1)
        embeddings_2 = list(request.sentences_2)

        output_1 = self.model.encode(embeddings_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        output_2 = self.model.encode(embeddings_2, return_dense=True, return_sparse=True, return_colbert_vecs=True)

        # Вычисление схожести для всех пар предложений
        similarity_scores = []
        for vec1 in output_1['colbert_vecs']:
            scores = []
            for vec2 in output_2['colbert_vecs']:
                score = self.model.colbert_score(vec1, vec2)
                scores.append(score)
            similarity_scores.append(scores)

        similarity_values = [score for sublist in similarity_scores for score in sublist]
        print("ColBERT similarity values:", similarity_values)

        response = SimilarityResponse(similarity=similarity_values)
        return response

