# triton-sandbox

A sandbox repository for studying Triton Inference Server features.

## Tracking by Detection

All stages of the algorithm are implemented as Triton models:
 - **Detection Preprocessing** (Python backend);
 - **Detection** (onnxruntime backend);
 - **Detection Postprocessing** (Python backend);
 - **Tracking** (Python backend).

The **Detection** model is based on [YOLOv9-c](https://github.com/WongKinYiu/yolov9) detector.

The **Tracking** model is [stateful](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#stateful-models) which allows it to differentiate inference requests from multiple clients using provided `CORRELATION ID`.

![Demo](https://github.com/voganesyan/triton-sandbox/blob/main/demo_gif/tracking-by-detection.gif)

Navigate to `tracking-by-detection` folder.
```bash
cd tracking-by-detection
```

### Launching Triton
Launch a `tritonserver` docker container.
```bash
docker run --gpus=all -it --shm-size=256m --rm    -p8000:8000 -p8001:8001 -p8002:8002   -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models   nvcr.io/nvidia/tritonserver:24.01-py3
```

Install dependencies for our Python backend scripts.
```bash
pip install -r /models/tracking/1/ocsort/requirements.txt
```

Launch Triton.
```bash
tritonserver --model-repository=/models
```

### Running Client
Run the client application.
```bash
python client.py --video test_data/MOT17-04-SDP-raw.webm
```

## References
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [BoxMOT](https://github.com/mikel-brostrom/yolo_tracking)
