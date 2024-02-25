# triton-sandbox

A sandbox repository for studying Triton Inference Server features.

Navigate to `model-ensemble` folder.
```bash
cd model-ensemble
```


## Launching Triton
Launch a `tritonserver` docker container.
```bash
docker run --gpus=all -it --shm-size=256m --rm    -p8000:8000 -p8001:8001 -p8002:8002   -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models   nvcr.io/nvidia/tritonserver:24.01-py3
```

Install dependencies for our Python backend scripts.
```bash
pip install opencv-python-headless
```

Launch Triton.
```bash
tritonserver --model-repository=/models
```

## Running Client
Run the client application.
```bash
python client.py
```
