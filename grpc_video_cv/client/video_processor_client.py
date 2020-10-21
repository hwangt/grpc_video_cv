from __future__ import print_function
import logging
import grpc
import numpy as np
import torch
import io
import cv2
import time

from grpc_video_cv.generated import video_processor_pb2_grpc, video_processor_pb2
import grpc_video_cv.client.camera as camera


class Client():
    def __init__(self, video, server, port):
        filename = None

        if video != '0':
            filename = video
            camera_id = 0
        else:
            camera_id = 0

        self.video_source = camera.VideoSource(camera_id=camera_id,
                                               size=(256, 256),
                                               filename=filename)

        # self.framegrabber = camera.VideoStream(video_source=self.video_source,
        #                                        fps=16)

        self.server = server
        self.port = port

    def gesture_request_builder(self, request_id, clip, width=256, height=256, count=4, ):
        request = video_processor_pb2.GestureRequest()
        request.request_id = request_id
        buf = io.BytesIO()
        torch.save(clip, buf)
        buf.seek(0)

        request.clip.width = width
        request.clip.height = height
        request.clip.count = count
        request.clip.bytes = buf.read()
        request.type = video_processor_pb2.GestureRequest.TWENTYBN_REALTIMENET
        return request

    def run(self, ):
        # Initialization of a few variables
        clip = np.random.randn(1, 4, 256, 256, 3)
        frame_index = 0
        clip_step_size = 4  # frames

        # Start threads
        # self.framegrabber.start()
        channel = grpc.insecure_channel(self.server + ':' + str(self.port))
        stub = video_processor_pb2_grpc.VideoStreamingProcessorStub(channel)

        while True:
            frame_index += 1

            # Grab frame if possible
            img_tuple = self.video_source.get_image()
            # If not possible, stop
            if img_tuple is None:
                break

            # Unpack
            img, numpy_img = img_tuple
            clip = np.roll(clip, -1, 1)
            clip[:, -1, :, :, :] = numpy_img

            # Preprocess
            new_clip = clip[:, :, :, ::-1].copy()
            new_clip /= 255.
            new_clip = new_clip.transpose(0, 1, 4, 2, 3)
            new_clip = torch.Tensor(new_clip).float()[0]

            if frame_index % 4 == 0:
                response = stub.Response(self.gesture_request_builder(request_id='request_id_' + str(frame_index),
                                                                      clip=new_clip))
                print(response)
                time.sleep(0.5)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="gRPC client to send a clip for gesture detection")
    parser.add_argument('--video', type=str, help='Video device ID or file', default='0')
    parser.add_argument('--server', type=str, help='Server address', default='localhost')
    parser.add_argument('--port', type=int, help='Port number', default=50051)
    args = parser.parse_args()
    logging.basicConfig()

    client = Client(video=args.video,
                    server=args.server,
                    port=args.port)

    client.run()
