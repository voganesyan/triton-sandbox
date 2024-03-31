import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from collections import defaultdict
import time


def draw_detections(img, detections: list, color=(255, 255, 255)):
    for detection in detections:
        x1, y1, x2, y2 = detection[:4].astype(int)
        score = detection[4]
        class_id = int(detection[5])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, f'{class_id}: {score:.2f}', (x1, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_track_histories(img, track_histories: list, color=(0, 255, 0)):
    track_lines = []
    for id, detections in track_histories.items():
        line = [[(det[0] + det[2]) // 2, det[3]] for det in detections]
        track_lines.append(np.array(line, np.int32))

        x1, y1, x2, y2 = detections[-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, str(id), (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    img = cv2.polylines(img, track_lines, False, color, 1)

def draw_fps(img, fps: int, color=(0, 0, 255)):
    cv2.putText(img, f'{fps} FPS', (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
def update_track_histories(tracks, track_histories):
    for track in tracks:
        x1, y1, x2, y2, id = track[:5].astype(int)
        if all(v == -1 for v in [x1, y1, x2, y2]):
            if id in track_histories:
                track_histories.pop(id)
        else:
            track_histories[id].append([x1, y1, x2, y2])


client = grpcclient.InferenceServerClient(url='localhost:8001')

cap = cv2.VideoCapture('test_data/MOT17-04-SDP-raw.webm')
if not cap.isOpened():
    print('Cannot open video')
    exit()

track_histories = defaultdict(list)
is_sequence_start = True
while True:
    ret, frame = cap.read()
    if not ret:
        print('Cannot receive frame (stream end?). Exiting...')
        break
    image_data = frame
    image_data = np.expand_dims(image_data, axis=0)
    input_tensor = grpcclient.InferInput('input_image',
                                         image_data.shape, 'UINT8')
    input_tensor.set_data_from_numpy(image_data)

    start_time = time.time()
    results = client.infer(model_name='tracking_by_detection', inputs=[input_tensor],
                           sequence_id=1, sequence_start=is_sequence_start)
    fps = int(1.0 / (time.time() - start_time))

    detections = results.as_numpy('detections')
    detections = np.squeeze(detections, axis=0)
    draw_detections(frame, detections)
    tracks = results.as_numpy('tracks')
    tracks = np.squeeze(tracks, axis=0)
    update_track_histories(tracks, track_histories)
    draw_track_histories(frame, track_histories)
    draw_fps(frame, fps)
    is_sequence_start = False
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
