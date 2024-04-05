import argparse
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi import AutoDetectionModel
from sahi.predict import predict

DEFAULT_MODEL_TYPE = "mmdet"
DEFAULT_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_DEVICE = "cuda:0"

DEFAULT_VISUAL_BBOX_THICKNESS = 2
DEFAULT_VISUAL_TEXT_SIZE = 2
DEFAULT_VISUAL_TEXT_THICKNESS = 1
DEFAULT_VISUAL_HIDE_LABELS = True

DEFAULT_MAX_DETECTIONS = [200, 250, 300]


def evaluate_dataset_split(
    model_path: str,
    config_path: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    dataset_split_path: str,
    dataset_annotations_path: str,
    output_directory_path: str,
) -> None:
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
    if not os.path.isdir(dataset_split_path):
        raise NotADirectoryError(
            f"The dataset split path '{dataset_split_path}' does not exist or is not a directory."
        )
    if not os.path.isfile(dataset_annotations_path):
        raise NotADirectoryError(
            f"The dataset annotations path '{dataset_annotations_path}' does not exist or is not a file."
        )

    os.makedirs(output_directory_path, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=DEFAULT_MODEL_TYPE,
        model_path=model_path,
        config_path=config_path,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        device=DEFAULT_DEVICE,
    )

    directory_name = "detections"

    detections_path = os.path.join(output_directory_path, directory_name, "result.json")

    _ = predict(
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        source=dataset_split_path,
        dataset_json_path=dataset_annotations_path,
        name=directory_name,
        project=output_directory_path,
        visual_bbox_thickness=DEFAULT_VISUAL_BBOX_THICKNESS,
        visual_text_size=DEFAULT_VISUAL_TEXT_SIZE,
        visual_text_thickness=DEFAULT_VISUAL_TEXT_THICKNESS,
        visual_hide_labels=DEFAULT_VISUAL_HIDE_LABELS,
    )

    coco_gt = COCO(dataset_annotations_path)
    coco_dt = coco_gt.loadRes(detections_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.params.maxDets = DEFAULT_MAX_DETECTIONS

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a MMDetection model on a dataset split."
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
        "--dataset_split_path",
        type=str,
        required=True,
        help="Path to the dataset split directory.",
    )
    parser.add_argument(
        "--dataset_annotations_path",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    parser.add_argument(
        "--output_directory_path",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    args = parser.parse_args()

    evaluate_dataset_split(
        model_path=args.model_path,
        config_path=args.config_path,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        dataset_split_path=args.dataset_split_path,
        dataset_annotations_path=args.dataset_annotations_path,
        output_directory_path=args.output_directory_path,
    )
