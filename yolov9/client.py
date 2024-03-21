import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from ocsort.ocsort import OCSort

def draw_detections(img, detections: list, color = (255, 255, 255)):
    for detection in detections:
        x1, y1, x2, y2 = detection[:4].astype(int)
        score = detection[4]
        class_id = int(detection[5])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, f'{class_id}: {score:.2f}', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_tracks(img, tracks: list, color = (0, 255, 0)):
    for track in tracks:
        x1, y1, x2, y2 = track[:4].astype(int)
        track_id = str(int(track[4]))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, track_id, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

client = grpcclient.InferenceServerClient(url='localhost:8001')

tracker = OCSort(
    per_class=True,
    det_thresh=0,
    max_age=30,
    min_hits=1,
    asso_threshold=0.3,
    delta_t=3,
    asso_func="giou",
    inertia=0.2,
    use_byte=False,
)

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
    detections = np.squeeze(detections, axis=0)
    tracks = tracker.update(detections, frame)
    draw_detections(frame, detections)
    draw_tracks(frame, tracks)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

