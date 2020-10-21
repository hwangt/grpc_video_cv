# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import video_processor_pb2 as video__processor__pb2


class VideoStreamingProcessorStub(object):
    """The greeting service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Response = channel.unary_unary(
            '/video_cv.VideoStreamingProcessor/Response',
            request_serializer=video__processor__pb2.GestureRequest.SerializeToString,
            response_deserializer=video__processor__pb2.GestureRequestResponse.FromString,
        )


class VideoStreamingProcessorServicer(object):
    """The greeting service definition.
    """

    def Response(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_VideoStreamingProcessorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Response': grpc.unary_unary_rpc_method_handler(
            servicer.Response,
            request_deserializer=video__processor__pb2.GestureRequest.FromString,
            response_serializer=video__processor__pb2.GestureRequestResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'video_cv.VideoStreamingProcessor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class VideoStreamingProcessor(object):
    """The greeting service definition.
    """

    @staticmethod
    def Response(request,
                 target,
                 options=(),
                 channel_credentials=None,
                 call_credentials=None,
                 insecure=False,
                 compression=None,
                 wait_for_ready=None,
                 timeout=None,
                 metadata=None):
        return grpc.experimental.unary_unary(request, target, '/video_cv.VideoStreamingProcessor/Response',
                                             video__processor__pb2.GestureRequest.SerializeToString,
                                             video__processor__pb2.GestureRequestResponse.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
