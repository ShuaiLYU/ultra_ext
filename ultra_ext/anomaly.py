from ultra_ext.im import collect_images

MVTEC_CATEGORIES = [
	"leather","grid","tile","wood","carpet",
	"cable","ze","pill","screw", # "toothbrush",
	"metal_nut","capsule","bottle","transistor","zipper"]



def get_mvtec_yolo_data(category,data_root="/Users/louis/workspace/ultra_louis_work/buffer/MVTEC/MVTec-YOLO"):
	"""Return train/test image paths for a given MVTec category."""

	assert category in MVTEC_CATEGORIES, f"Unknown category: {category}"
	train_im_dir = f"{data_root}/{category}/train/good"
	test_im_dir = f"{data_root}/{category}/test"
	test_good_im_dir = f"{test_im_dir}/good"
	train_im_list=collect_images(train_im_dir,recursively=True)
	test_im_list=collect_images(test_im_dir,recursively=True)
	test_good_im_list=collect_images(test_good_im_dir,recursively=True)

	test_anomaly_im_list = [im for im in test_im_list if im not in test_good_im_list]	

	return dict(
		train_im_dir=train_im_dir,
		test_im_dir=test_im_dir,
		train_im_list=train_im_list,
		test_im_list=test_im_list,
		test_good_im_list=test_good_im_list,
		test_anomaly_im_list=test_anomaly_im_list,

		data_yaml=data_root+f"/{category}.yaml",
	)


"""Quick probe — print backbone-feature shapes vs. truncated-cls-head feature shapes."""
import torch
from ultralytics.models.yolo.model import YOLOAnomaly


def probe_channels(model_path: str, imgsz: int = 640) -> dict:
    """Print and return backbone vs. cls-head feature shapes for each detection level.

    Args:
        model_path: Path to a YOLOAnomaly .pt checkpoint.
        imgsz: Square inference size (default 640).

    Returns:
        dict with keys 'x_shapes' and 'cls_feats_shapes', each a list of tuples.
    """
    m = YOLOAnomaly(model_path)
    head = m.model.model[-1]
    _, cls_heads = head._get_feature_heads()

    captured = {}
    orig_forward = head.forward

    def hook(x):
        captured["x_shapes"] = [tuple(t.shape) for t in x]
        captured["cls_feats_shapes"] = [tuple(cls_heads[i](x[i]).shape) for i in range(head.nl)]
        return orig_forward(x)

    head.forward = hook
    with torch.no_grad():
        m.model(torch.zeros(1, 3, imgsz, imgsz))
    head.forward = orig_forward  # restore

    print(f"\n[{model_path}]  imgsz={imgsz}")
    print("  backbone features (x[i]):")
    for i, s in enumerate(captured["x_shapes"]):
        print(f"    layer {i}: {s}")
    print("  truncated cls_head outputs (cls_heads[i](x[i])):")
    for i, s in enumerate(captured["cls_feats_shapes"]):
        print(f"    layer {i}: {s}")

    return captured


# if __name__ == "__main__":
#     probe_channels("./runs/temp/bottle_yolo26l_anomaly_model.pt")