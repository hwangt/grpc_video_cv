from concurrent import futures
from grpc_video_cv.generated import video_processor_pb2_grpc
from grpc_video_cv.server.grpc_video_processor import VideoProcessor

import grpc


class Server:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        video_processor_pb2_grpc.add_VideoStreamingProcessorServicer_to_server(VideoProcessor(self.checkpoint_dir),
                                                                               server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="gRPC server to detect hand gestures")
    parser.add_argument('--checkpoint_dir', type=str, help='Path to model checkpoints', required=False)
    args = parser.parse_args()

    server = Server(checkpoint_dir=args.checkpoint_dir)
    server.run()
