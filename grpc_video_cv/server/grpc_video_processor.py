from grpc_video_cv.generated import video_processor_pb2_grpc, video_processor_pb2
from grpc_video_cv.generated.video_processor_pb2_grpc import VideoStreamingProcessorServicer
import torch
from grpc_video_cv.model import feature_extractors, nn_utils
import io
import os


class VideoProcessor(VideoStreamingProcessorServicer):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

        # Load feature extractor
        self.feature_extractor = feature_extractors.StridedInflatedEfficientNet()
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'strided_inflated_efficientnet.ckpt'))
        self.feature_extractor.load_state_dict(checkpoint)
        self.feature_extractor.eval()

        # Load a logistic regression classifier
        self.gesture_classifier = nn_utils.LogisticRegression(num_in=self.feature_extractor.feature_dim,
                                                              num_out=30)
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'efficientnet_logistic_regression.ckpt'))
        self.gesture_classifier.load_state_dict(checkpoint)
        self.gesture_classifier.eval()

        self.gesture_model = nn_utils.Pipe(self.feature_extractor, self.gesture_classifier)
        self.gesture_model.eval()

    def extract_gesture(self, request, response):
        # Create input clip tensor from request
        buffer = io.BytesIO(request.clip.bytes)
        clip_torch = torch.load(buffer)
        clip_torch = clip_torch.view(request.clip.count, 3, request.clip.height,
                                     request.clip.width)  # e.g. 4, 3, 256, 256

        # Predict gesture from clip (get softmax)
        predictions = self.gesture_model(clip_torch)
        if isinstance(predictions, list):
            predictions = [pred.cpu().detach().numpy()[0] for pred in predictions]
        else:
            predictions = predictions.cpu().detach().numpy()[0]

        # Sort highest conf to least
        sorted_indices = predictions.argsort()[::-1]
        return sorted_indices, predictions

    def Response(self, request, context):
        print(f'Received {request.request_id}')
        response = video_processor_pb2.GestureRequestResponse()
        response.request_id = request.request_id
        response.response_id = 'response_id1'  # TODO: define policy for generating IDs
        sorted_indices, predictions = self.extract_gesture(request, response)

        # If the highest confidence gesture pred is > 0.2, then we succeeded
        if predictions[sorted_indices[0]] > 0.2:
            response.status = video_processor_pb2.GestureRequestResponse.SUCCESS
            # response.gesture = video_processor_pb2.GestureRequestResponse.Gesture.Name(sorted_indices[0])
            response.gesture = sorted_indices[0]
        else:
            response.status = video_processor_pb2.GestureRequestResponse.FAILURE

        return response
