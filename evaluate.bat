python mmdetection/tools/test.py configs/rtmdet-tiny_full_4544.py models/rtmdet-tiny_full_4544.pth

python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-tiny_sahi_640.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_640.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-tiny_sahi_640_val"
	
python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-tiny_sahi_640_val/detections/result.json"
	
python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-tiny_sahi_640_aug.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_640_aug.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-tiny_sahi_640_aug_val"
	
python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-tiny_sahi_640_aug_val/detections/result.json"
	
python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-s_sahi_640.pth" ^
    --config_path "configs/rtmdet-s_sahi_640.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-s_sahi_640_val"
	
python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-s_sahi_640_val/detections/result.json"
	
python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-s_sahi_640_aug.pth" ^
    --config_path "configs/rtmdet-s_sahi_640_aug.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-s_sahi_640_aug_val"

python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-s_sahi_640_aug_val/detections/result.json"
	
python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-tiny_sahi_1088.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_1088.py" ^
    --slice_height 1088 ^
    --slice_width 1088 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-tiny_sahi_1088_val"
	
python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-tiny_sahi_1088_val/detections/result.json"

python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-tiny_sahi_1088_aug.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_1088_aug.py" ^
    --slice_height 1088 ^
    --slice_width 1088 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-tiny_sahi_1088_aug_val"

python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-tiny_sahi_1088_aug_val/detections/result.json"

python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-s_sahi_1088.pth" ^
    --config_path "configs/rtmdet-s_sahi_1088.py" ^
    --slice_height 1088 ^
    --slice_width 1088 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-s_sahi_1088_val"

python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-s_sahi_1088_val/detections/result.json"

python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-s_sahi_1088_aug.pth" ^
    --config_path "configs/rtmdet-s_sahi_1088_aug.py" ^
    --slice_height 1088 ^
    --slice_width 1088 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/val" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/val/val-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-s_sahi_1088_aug_val"

python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/val/val-annotations.json" "evaluation-output/rtmdet-s_sahi_1088_aug_val/detections/result.json"

:: Best model test evaluation

python scripts/sahi_evaluate.py ^
    --model_path "models/rtmdet-tiny_sahi_640.pth" ^
    --config_path "configs/rtmdet-tiny_sahi_640.py" ^
    --slice_height 640 ^
    --slice_width 640 ^
    --overlap_height_ratio 0.1 ^
    --overlap_width_ratio 0.1 ^
    --dataset_split_path "data/sunflower_dataset_v2.0/test" ^
	--dataset_annotations_path "data/sunflower_dataset_v2.0/test/test-annotations.json" ^
    --output_directory_path "evaluation-output/rtmdet-tiny_sahi_640_test"
	
python scripts/evaluate_defective_ratio.py "data/sunflower_dataset_v2.0/test/test-annotations.json" "evaluation-output/rtmdet-tiny_sahi_640_test/detections/result.json"