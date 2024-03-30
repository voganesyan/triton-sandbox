import json
import numpy as np
from ocsort.ocsort import OCSort
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        print('Tracking: initialization...')
        model_config = json.loads(args['model_config'])
        tracks = pb_utils.get_output_config_by_name(
            model_config, 'tracks'
        )
        self.tracks_dtype = pb_utils.triton_string_to_numpy(
            tracks['data_type']
        )
        self.tracker = OCSort(
            per_class=True,
            det_thresh=0,
            max_age=30,
            min_hits=1,
            asso_threshold=0.3,
            delta_t=3,
            asso_func='giou',
            inertia=0.2,
            use_byte=False,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            detections = pb_utils.get_input_tensor_by_name(
                request, 'detections'
            )
            frame = pb_utils.get_input_tensor_by_name(
                request, 'frame'
            )
            detections = np.squeeze(detections.as_numpy(), axis=0)
            frame = np.squeeze(frame.as_numpy(), axis=0)

            tracks = self.tracker.update(detections, frame)
            tracks = np.expand_dims(tracks, axis=0)
            tracks = pb_utils.Tensor(
                'tracks', tracks.astype(self.tracks_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tracks]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Tracking: cleaning up...')
