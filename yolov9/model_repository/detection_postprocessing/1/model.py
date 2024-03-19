import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "detection_postprocessing_output"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        MODEL_IMAGE_SIZE = (640, 640)

        def xywh2xyxy(x):
            # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
            x1 = x[0] - x[2] / 2
            y1 = x[1] - x[3] / 2
            x2 = x[0] + x[2] / 2
            y2 = x[1] + x[3] / 2
            return [x1, y1, x2, y2]

        def postprocess(outputs, image_shape, conf_thresold = 0.4, iou_threshold = 0.4):
            predictions = np.squeeze(outputs).T
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > conf_thresold, :]
            scores = scores[scores > conf_thresold]
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            boxes = predictions[:, :4]
            boxes = np.array(boxes)            
            boxes[:, 0::2] *= image_shape[1] / MODEL_IMAGE_SIZE[1]
            boxes[:, 1::2] *= image_shape[0] / MODEL_IMAGE_SIZE[0]
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresold, iou_threshold)
            detections = [np.append(
                xywh2xyxy(boxes[i]), [scores[i], class_ids[i]]) for i in indices] 
            return np.array(detections)

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input"
            )
            proc_params = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_params"
            )

            detections = np.squeeze(in_0.as_numpy(), axis=0)
            proc_params = np.squeeze(proc_params.as_numpy(), axis=0)
            detections = postprocess(detections, proc_params)
            
            input_tensor = np.expand_dims(detections, axis=0)
            out_tensor_0 = pb_utils.Tensor(
                "detection_postprocessing_output", input_tensor.astype(output0_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
