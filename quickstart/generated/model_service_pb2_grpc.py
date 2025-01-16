# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import model_service_pb2 as model__service__pb2

GRPC_GENERATED_VERSION = '1.68.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in model_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class SimilarityServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ComputeSimilarity = channel.unary_unary(
                '/similarity.SimilarityService/ComputeSimilarity',
                request_serializer=model__service__pb2.SimilarityRequest.SerializeToString,
                response_deserializer=model__service__pb2.SimilarityResponse.FromString,
                _registered_method=True)
        self.SparseEmbedding = channel.unary_unary(
                '/similarity.SimilarityService/SparseEmbedding',
                request_serializer=model__service__pb2.SimilarityRequest.SerializeToString,
                response_deserializer=model__service__pb2.SimilarityResponse.FromString,
                _registered_method=True)
        self.ColBERT = channel.unary_unary(
                '/similarity.SimilarityService/ColBERT',
                request_serializer=model__service__pb2.SimilarityRequest.SerializeToString,
                response_deserializer=model__service__pb2.SimilarityResponse.FromString,
                _registered_method=True)


class SimilarityServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ComputeSimilarity(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SparseEmbedding(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ColBERT(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimilarityServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ComputeSimilarity': grpc.unary_unary_rpc_method_handler(
                    servicer.ComputeSimilarity,
                    request_deserializer=model__service__pb2.SimilarityRequest.FromString,
                    response_serializer=model__service__pb2.SimilarityResponse.SerializeToString,
            ),
            'SparseEmbedding': grpc.unary_unary_rpc_method_handler(
                    servicer.SparseEmbedding,
                    request_deserializer=model__service__pb2.SimilarityRequest.FromString,
                    response_serializer=model__service__pb2.SimilarityResponse.SerializeToString,
            ),
            'ColBERT': grpc.unary_unary_rpc_method_handler(
                    servicer.ColBERT,
                    request_deserializer=model__service__pb2.SimilarityRequest.FromString,
                    response_serializer=model__service__pb2.SimilarityResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'similarity.SimilarityService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('similarity.SimilarityService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class SimilarityService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ComputeSimilarity(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/similarity.SimilarityService/ComputeSimilarity',
            model__service__pb2.SimilarityRequest.SerializeToString,
            model__service__pb2.SimilarityResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SparseEmbedding(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/similarity.SimilarityService/SparseEmbedding',
            model__service__pb2.SimilarityRequest.SerializeToString,
            model__service__pb2.SimilarityResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ColBERT(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/similarity.SimilarityService/ColBERT',
            model__service__pb2.SimilarityRequest.SerializeToString,
            model__service__pb2.SimilarityResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
