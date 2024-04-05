![sample](images/visuals.png)

# Agri-det

At the moment, this library facilitates the identification of defective sunflowers using aerial images. It is planned to be expanded for use in UAV-based agricultural computer vision applications.

## Prerequisites

### Hardware

- Requires an NVIDIA GPU
- Minimum of 16 GB VRAM needed
- At least 32 GB of RAM recommended

### Installation

- [CUDA](https://developer.nvidia.com/cuda-downloads) and [Python](https://www.python.org/downloads/) are required

- Install [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html) following the official instructions for the GPU platforms

- Install [SAHI](https://github.com/obss/sahi?tab=readme-ov-file#installation)

## Data

Data is only required if you want to reproduce the training.

### Download data

Defective sunflower dataset can be downloaded from the [GraviLink cloud](https://cloud.gravilink.org/s/DfcQYqxwzCYWN6f). The dataset folder should be placed in the `data` folder of the repository: `agri-det/data/sunflower_dataset_v1.0`.

### Prepare data

To slice the dataset using SAHI, run the following script:

```
prepare_data.bat
```

## Train

### Reproduce all model training

To reproduce the training of all provided models, run the following script:

```
train.bat
```

### Train a specific model

To train a specific model using the provided configuration, run the following command, replacing the `config.py` with the desired model config:

```
python mmdetection/tools/train.py configs/config.py
```

## Evaluate

### Download trained model files

Model files can be downloaded from the [GraviLink cloud](https://cloud.gravilink.org/s/NaimEt7oKQDfrwt). Model files should be placed in the `models` folder of the repository: `agri-det/models`.

### Evaluate all trained models

To evaluate all the trained models, run the following script:

```
evaluate.bat
```

### Evaluate a specific model

To evaluate a non-SAHI model, run the following script, replacing the `config.py` with the desired model config and `model.pth` with the corresponding model file:

```
python mmdetection/tools/test.py configs/config.py models/model.pth
```

To evaluate a SAHI model, run the `scripts/sahi_evaluate.py` script. For example:

```
python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-tiny_sahi_640.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_640.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v1.0/val" ^
    --dataset_annotations_path "data/sunflower_dataset_v1.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-tiny_sahi_640"
```

## Inference

To inference with a non-SAHI model, run the `mmdetection/demo/image_demo.py` script with the desired parameters. For example:

```
python mmdetection/demo/image_demo.py ^
    data/sunflower_dataset_v1.0/test/images/56c44d14-DJI_0031.JPG ^
    configs/rtmdet-tiny_sahi_640.py ^
    --weights models/rtmdet-tiny_sahi_640.pth
```

To perform SAHI model inference, use the `scripts/sahi_inference.py` script. For example:

```
python inference_single_image.py ^
    --model_path "models/rtmdet-tiny_sahi_640.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_640.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --input_image_path "data/sunflower_dataset_v1.0/test/images/56c44d14-DJI_0031.JPG" ^
    --output_directory_path "inference-output"
```