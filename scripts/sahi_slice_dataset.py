import argparse
import os

from sahi.slicing import slice_coco

DEFAULT_IGNORE_NEGATIVE_SAMPLES = False
DEFAULT_MIN_AREA_RATIO = 0
DEFAULT_VERBOSE = 1


def slice_split(
    dataset_root_directory_path: str,
    output_root_directory_path: str,
    split: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
):
    if overlap_height_ratio < 0 or overlap_height_ratio >= 1:
        raise ValueError("Overlap height ratio must be in [0; 1) range.")
    if overlap_width_ratio < 0 or overlap_width_ratio >= 1:
        raise ValueError("Overlap width ratio must be in [0; 1) range.")

    original_coco_annotation_path = os.path.join(
        dataset_root_directory_path, split, f"{split}-annotations.json"
    )

    image_dir = os.path.join(dataset_root_directory_path, split)
    output_dir = os.path.join(output_root_directory_path, f"{split}/")

    os.makedirs(output_dir, exist_ok=True)

    slice_coco(
        coco_annotation_file_path=original_coco_annotation_path,
        image_dir=image_dir,
        output_coco_annotation_file_name=f"{split}-annotations.json",
        output_dir=output_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        ignore_negative_samples=DEFAULT_IGNORE_NEGATIVE_SAMPLES,
        min_area_ratio=DEFAULT_MIN_AREA_RATIO,
        verbose=DEFAULT_VERBOSE,
    )


def main(
    dataset_root_directory_path: str,
    output_root_directory_path: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
):
    if not os.path.exists(dataset_root_directory_path):
        raise (
            f"The dataset root directory path '{dataset_root_directory_path}' does not exist."
        )

    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_root_directory_path, split)

        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"The required dataset split '{split}' does not exist at '{split_path}'."
            )

        if not os.path.exists(os.path.join(split_path, f"{split}-annotations.json")):
            raise FileNotFoundError(
                f"The COCO annotation file for the '{split}' split does not exist."
            )

        slice_split(
            dataset_root_directory_path=dataset_root_directory_path,
            output_root_directory_path=output_root_directory_path,
            split=split,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice a COCO dataset.")
    parser.add_argument(
        "dataset_root_directory_path",
        type=str,
        required=True,
        help="Path to the root directory of the original dataset.",
    )
    parser.add_argument(
        "output_root_directory_path",
        type=str,
        required=True,
        help="Path to the root directory for the output sliced dataset.",
    )
    parser.add_argument(
        "slice_height", type=int, required=True, help="Height of the slices."
    )
    parser.add_argument(
        "slice_width", type=int, required=True, help="Width of the slices."
    )
    parser.add_argument(
        "overlap_height_ratio",
        type=float,
        required=True,
        help="Slice overlap ratio for the height.",
    )
    parser.add_argument(
        "overlap_width_ratio",
        type=float,
        required=True,
        help="Slice overlap ratio for the width.",
    )

    args = parser.parse_args()

    main(
        args.dataset_root_directory_path,
        args.output_root_directory_path,
        args.slice_height,
        args.slice_width,
        args.overlap_height_ratio,
        args.overlap_width_ratio,
    )
