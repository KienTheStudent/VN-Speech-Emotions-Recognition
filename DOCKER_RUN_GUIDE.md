# GPU Docker Guide (Speech-Emotions-Recognition)

This guide shows how to build, run, and reuse the GPU container.

## 1) Prerequisites (host machine)

- Docker installed
- NVIDIA GPU driver installed
- NVIDIA Container Toolkit installed

Quick GPU check on host:

```bash
nvidia-smi
```

## 2) Build image (from project root)

```bash
docker build -t ser-gpu:latest .
```

## 3) First run (create container with GPU)

```bash
docker run --gpus all -it --name ser-gpu-container \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ser-gpu:latest
```

What this does:
- `--gpus all`: exposes all GPUs to container
- `--name ser-gpu-container`: gives stable container name
- `-v "$(pwd)":/workspace`: mounts your current project folder

## 4) Verify GPU inside container

Run inside container shell:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count()); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 5) Train or predict inside container

Examples:

```bash
python ECAPA/train_emotion_model.py
python ECAPA/predict_emotion.py your_audio.wav --model_dir ECAPA/emotion_model

python "DFAT_Hybrid_Fusion/train_dualstream.py"
python "DFAT_Hybrid_Fusion/predict_dualstream.py" your_audio.wav --model_dir "DFAT_Hybrid_Fusion/dualstream_model"
```

## 6) Exit and reuse the same container

Exit container shell:

```bash
exit
```

Start existing container again:

```bash
docker start -ai ser-gpu-container
```

Or run a one-off command in existing container:

```bash
docker exec -it ser-gpu-container bash
```

## 7) Optional cleanup

```bash
docker rm -f ser-gpu-container
docker rmi ser-gpu:latest
```
