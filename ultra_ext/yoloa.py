
# for YOLOAnomaly 

import csv
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from ultra_ext.utils import collect_images

MVTEC_CATEGORIES: list[str] = [
    "leather", "grid", "tile", "wood", "carpet",
    "cable", "hazelnut", "pill", "screw",
    "metal_nut", "capsule", "bottle", "transistor", "zipper",
]
BTAD_CATEGORIES: list[str] = [ "01","02","03"]


WEIGHTS_BANK= {
		"yoloe26n_mvtec_defect": "./runs/yoloe27_tp/26n_ptwobjv1_bs256_epo10_close2_old_mvtec_yolo_merge1/weights/best.pt",
        "yoloe26n": "yoloe-26n-seg.pt",
        "yoloe26l": "yoloe-26l-seg.pt",
        "yolo26n": "yolo26n.pt",
        "yolo26l": "yolo26l.pt",
	}





def get_arguments(category: str = "screw") -> tuple[dict, dict]:
    """Return default (model_arg, anomaly_arg) inference dicts.

    Args:
        category: MVTec category name (unused currently, reserved for future
            per-category tuning).

    Returns:
        Tuple of (model_arg, anomaly_arg) dicts.
    """
    model_arg = dict(conf=0.1, iou=0.25, max_det=1000, imgsz=640, single_cls=True, rect=False)
    anomaly_arg = dict(
        
        mode="anomaly",
        score_filter_kernel=1,
        active_layers=[0,1, 2],

        # for bank building and calibration
        auto_temperature=True,
        em_iters=5,
        accumulate_thresh=0.3,
        calibration_interval=1,
        calibration_target_score=0.2,
        
        # ------- for infer 
        ad_conf=0.4,
        ad_max_det=10,
        # ------------ for heatmpa 
        return_heatmap=True,
        feature_mode="fused_heatmap", 
		fused_use_pre_clshead=True, 
        fused_layers=[0]
    )
    return model_arg, anomaly_arg




import os 
def get_mvtecad_mask(im_file: str) -> str | None:
    """Return the MVTec ground-truth mask path for *im_file*, or None if absent.

    Converts ``…/test/<defect>/image.png`` →
    ``…/ground_truth/<defect>/image_mask.png``.
    """
    mask = im_file.replace("test", "ground_truth").replace(".png", "_mask.png")
    return mask if os.path.exists(mask) else None





def get_mvtec_yolo_data(
    category: str,
    data_root: str = "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO",
) -> dict[str, Any]:
    """Return train/test image paths and YAML config for one MVTec category.

    Args:
        category: One of :data:`MVTEC_CATEGORIES`.
        data_root: Root directory of the MVTec-YOLO dataset.

    Returns:
        Dict with keys ``train_im_dir``, ``test_im_dir``, ``train_im_list``,
        ``train_im10`` (first 10 train images for quick smoke tests),
        ``test_im_list``, ``test_good_im_list``, ``test_anomaly_im_list``,
        ``data_yaml``.
    """
    

    if category in MVTEC_CATEGORIES:
        data_root = "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO"
        train_im_dir = f"{data_root}/{category}/train/good"
        test_im_dir = f"{data_root}/{category}/test"
        test_good_im_dir = f"{test_im_dir}/good"
    elif category in BTAD_CATEGORIES:
        data_root = "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/BTAD/BTech_Dataset_transformed"
        train_im_dir = f"{data_root}/{category}/train/ok"
        test_im_dir = f"{data_root}/{category}/test"
        test_good_im_dir = f"{test_im_dir}/ok"
    else:
        raise ValueError(f"Unknown category '{category}'. Must be one of {MVTEC_CATEGORIES} or {BTAD_CATEGORIES}.")
 


    train_im_list = collect_images(train_im_dir, recursively=True)
    test_im_list = collect_images(test_im_dir, recursively=True)
    test_good_im_list = collect_images(test_good_im_dir, recursively=True)
    test_anomaly_im_list = [im for im in test_im_list if im not in test_good_im_list]
    return dict(
        train_im_dir=train_im_dir,
        test_im_dir=test_im_dir,
        train_im_list=train_im_list,
        train_im10=train_im_list[:10],
        test_im_list=test_im_list,
        test_good_im_list=test_good_im_list,
        test_anomaly_im_list=test_anomaly_im_list,
        data_yaml=f"{data_root}/{category}/{category}.yaml",
    )





YOLOA_BUILD_DEFAULTS: dict[str, Any] = dict(
	feature_mode="fused_heatmap",
	return_heatmap=True,
	fused_use_pre_clshead=True,
	fused_layers=[0],        # P3 only, 80×80
)




def build_yolo_anomaly(
	base_model: str | Path,
	support_imgs: Sequence[str | Path],
	*,
	imgsz: int = 640,
	anomaly_arg: dict | None = None,
	yolo_weight: float = 0.0,
	build_overrides: dict | None = None,
	verbose: bool = False,
):
	"""Build, configure, and bank a YOLOAnomaly model.

	Args:
		base_model: Path to a YOLO/YOLOE checkpoint.
		support_imgs: List of normal-image paths to seed the memory bank.
		imgsz: Inference resolution used both during bank build and val/predict.
		anomaly_arg: Per-category anomaly knobs from ``get_arguments(category)``
			(``accumulate_thresh``, ``ad_conf``, ``em_iters``, …). May be ``None``.
		yolo_weight: 0.0 = pure memory-bank cosine score (default).  > 0 blends the
			YOLO classifier-head logit into the anomaly score.
		build_overrides: Optional dict that wins over ``YOLOA_BUILD_DEFAULTS``.
			Use to flip ``feature_mode``, ``fused_layers``, etc., per-experiment.
		verbose: Forward to ``load_support_set``.

	Returns:
		YOLOAnomaly: Ready for ``.predict()`` / ``.val()``.
	"""
	from ultralytics.models.yolo.model import YOLOAnomaly

	model = YOLOAnomaly(base_model)
	model.setup(["anomaly"])
	if anomaly_arg:
		model.set_anomaly_args(**anomaly_arg)
	if yolo_weight > 0:
		model.set_anomaly_args(yolo_weight=yolo_weight)
	# Merge per-experiment overrides on top of the defaults; the result is the
	# *inference-time* config (feature_mode etc.) — applied AFTER banking.
	build_kw = {**YOLOA_BUILD_DEFAULTS, **(build_overrides or {})}
	inference_feature_mode = build_kw.pop("feature_mode", "fused_heatmap")
	# CRITICAL: the OBMA bank MUST be built in feature_mode='per_level' so the four
	# per-head banks (used by the bbox-decode branch) are seeded.  Building in
	# 'fused_heatmap' mode populates only the fused P3 bank and starves the
	# per-level heads → bbox-decode produces almost no detections.  Once the bank
	# is frozen, switching feature_mode is just an inference-time toggle.
	model.set_anomaly_args(feature_mode="per_level", **build_kw)
	model.load_support_set(support_imgs, imgsz=imgsz, verbose=verbose)
	# Restore the user-requested inference mode (defaults to fused_heatmap so
	# pixel-AUROC and connected-component boxes work out of the box).
	model.set_anomaly_args(feature_mode=inference_feature_mode)
	return model
