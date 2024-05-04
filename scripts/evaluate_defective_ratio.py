import argparse
import json
import os

import numpy as np

DEFAULT_SCORE_THRESHOLD = 0.4


def load_ground_truth_annotations(file_path):
    with open(file_path) as f:
        data = json.load(f)
    image_annotations = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category = ann["category_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = {"total": 0, "negative": 0}
        image_annotations[image_id]["total"] += 1
        if category == 0:
            image_annotations[image_id]["negative"] += 1
    ratios = {
        img_id: counts["negative"] / counts["total"] if counts["total"] > 0 else 0
        for img_id, counts in image_annotations.items()
    }
    return ratios, image_annotations


def load_predictions_and_calculate_stats(predictions_file, gt_ratios):
    errors = []
    detailed_stats = {}
    if os.path.exists(predictions_file):
        with open(predictions_file) as f:
            predictions = json.load(f)
        image_predictions = {}
        for pred in predictions:
            image_id = pred["image_id"]
            score = pred["score"]
            if score >= DEFAULT_SCORE_THRESHOLD:
                if image_id not in image_predictions:
                    image_predictions[image_id] = {"total": 0, "negative": 0}
                image_predictions[image_id]["total"] += 1
                if pred["category_id"] == 0:
                    image_predictions[image_id]["negative"] += 1
        for img_id, counts in image_predictions.items():
            predicted_ratio = (
                counts["negative"] / counts["total"] if counts["total"] > 0 else 0
            )
            if img_id in gt_ratios:
                errors.append(abs(predicted_ratio - gt_ratios[img_id]))
                detailed_stats[img_id] = {
                    "total_detections": counts["total"],
                    "negative_detections": counts["negative"],
                    "predicted_ratio": predicted_ratio,
                    "gt_ratio": gt_ratios[img_id],
                }
    mean_absolute_error = np.mean(errors) if errors else 0
    return mean_absolute_error, detailed_stats


def evaluate_experiment(gt_file, predictions_file):
    gt_ratios, object_counts = load_ground_truth_annotations(gt_file)
    mean_absolute_error, detailed_stats = load_predictions_and_calculate_stats(
        predictions_file, gt_ratios
    )
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    for img_id, stats in detailed_stats.items():
        print(
            f"Image ID: {img_id:10} | Total Detections: {stats['total_detections']:4} | Negative Detections: {stats['negative_detections']:4} | Predicted Ratio: {stats['predicted_ratio']:4.3f} | GT Ratio: {stats['gt_ratio']:4.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a single experiment against ground truth data."
    )
    parser.add_argument(
        "gt_file",
        type=str,
        help="Path to the JSON file containing ground truth annotations.",
    )
    parser.add_argument(
        "predictions_file",
        type=str,
        help="Path to the JSON file containing prediction results.",
    )

    args = parser.parse_args()

    evaluate_experiment(args.gt_file, args.predictions_file)
