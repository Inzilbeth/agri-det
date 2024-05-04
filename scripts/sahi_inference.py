import argparse
import os

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil

DEFAULT_MODEL_TYPE = "mmdet"
DEFAULT_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_DEVICE = "cuda:0"

DEFAULT_TEXT_SIZE = 2
DEFAULT_RECT_TH = 2
DEFAULT_HIDE_LABELS = True

DEFAULT_MATCH_METRIC = "IOU"
DEFAULT_MATCH_THRESHOLD = 0.35


def inference(
    model_path: str,
    config_path: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    input_image_path: str,
    output_directory_path: str,
):
    if not os.path.isfile(model_path):
        raise NotADirectoryError(
            f"The model weights path '{model_path}' does not exist or is not a file."
        )
    if not os.path.isfile(config_path):
        raise NotADirectoryError(
            f"The model config path '{config_path}' does not exist or is not a file."
        )
    if overlap_height_ratio < 0 or overlap_height_ratio >= 1:
        raise ValueError("Overlap height ratio must be in [0; 1) range.")
    if overlap_width_ratio < 0 or overlap_width_ratio >= 1:
        raise ValueError("Overlap width ratio must be in [0; 1) range.")
    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(
            f"The input image path '{input_image_path}' is not found."
        )

    os.makedirs(output_directory_path, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=DEFAULT_MODEL_TYPE,
        model_path=model_path,
        config_path=config_path,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        device=DEFAULT_DEVICE,
    )

    image = read_image_as_pil(input_image_path)

    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        postprocess_match_metric=DEFAULT_MATCH_METRIC,
        postprocess_match_threshold=0.35,
    )

    result.export_visuals(
        export_dir=output_directory_path,
        text_size=DEFAULT_TEXT_SIZE,
        rect_th=DEFAULT_RECT_TH,
        hide_labels=DEFAULT_HIDE_LABELS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a single image using a MMDetection model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the MMDetection model weights file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the MMDetection model config file.",
    )
    parser.add_argument(
        "--slice_height", type=int, required=True, help="Height of the slices."
    )
    parser.add_argument(
        "--slice_width", type=int, required=True, help="Width of the slices."
    )
    parser.add_argument(
        "--overlap_height_ratio",
        type=float,
        required=True,
        help="Slice overlap ratio for the height.",
    )
    parser.add_argument(
        "--overlap_width_ratio",
        type=float,
        required=True,
        help="Slice overlap ratio for the width.",
    )
    parser.add_argument(
        "--input_image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--output_directory_path",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    args = parser.parse_args()

    inference(
        args.model_path,
        args.config_path,
        args.slice_height,
        args.slice_width,
        args.overlap_height_ratio,
        args.overlap_width_ratio,
        args.input_image_path,
        args.output_directory_path,
    )
