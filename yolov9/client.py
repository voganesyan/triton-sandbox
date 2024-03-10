import cv2
import numpy as np
import tritonclient.grpc as grpcclient

MODEL_IMAGE_SIZE = (640, 640)


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y 

def postprocess(outputs, image_shape, conf_thresold = 0.4, iou_threshold = 0.4):
    predictions = np.squeeze(outputs).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    boxes = predictions[:, :4]
    
    input_shape = np.array([MODEL_IMAGE_SIZE[0], MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], MODEL_IMAGE_SIZE[1]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
    boxes = boxes.astype(np.int32)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresold, iou_threshold)
    detections = []
    for index in indices:
        detections.append({
            'class_id': class_ids[index],
            'score': scores[index],
            'box': xywh2xyxy(boxes[index])
        })
    return detections

def draw_detections(img, detections: list, color = (0, 255, 0)):
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        class_id = detection['class_id']
        score = detection['score']

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{class_id}: {score:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


client = grpcclient.InferenceServerClient(url='localhost:8001')

cap = cv2.VideoCapture('test_data/MOT17-04-SDP-raw.webm')
if not cap.isOpened():
    print('Cannot open video')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cannot receive frame (stream end?). Exiting...')
        break
    image_data = frame
    image_data = np.expand_dims(image_data, axis=0)
    input_tensor = grpcclient.InferInput('input_image', image_data.shape, 'UINT8')
    input_tensor.set_data_from_numpy(image_data)
    
    results = client.infer(model_name='ensemble_model', inputs=[input_tensor])
    detections = results.as_numpy('detections')
    detections = postprocess(detections, frame.shape)
    draw_detections(frame, detections)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

