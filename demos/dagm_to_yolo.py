"""Demo: convert the DAGM dataset (Class1..Class10) into YOLO format."""

import shutil
from pathlib import Path

from ultra_ext import DagmFolder


def main():
    src = Path("~/workspace/ultra_louis_work/buffer/AnomalyData/DAGM").expanduser()
    dst = Path("~/workspace/ultra_louis_work/buffer/AnomalyData/DAGM_yolo").expanduser()

    folder = DagmFolder(src)
    print(f"loaded {len(folder)} images from {src}")
    print(f"defect counts (per class): {dict(folder.class_counts())}")

    yaml_paths = folder.to_yolo_per_class(
        out_dir=dst,
        train_ratio=0.9,
        seed=42,
        link_mode="copy",
        include_normal=True,
        class_name="anomaly",
    )
    print(f"done -> {len(yaml_paths)} per-class datasets under {dst}\n")

    print("zipping per-class datasets ...")
    for yaml_path in yaml_paths:
        cls_dir = yaml_path.parent
        zip_base = dst / f"DAGM_{cls_dir.name}"
        archive = shutil.make_archive(str(zip_base), "zip", root_dir=dst, base_dir=cls_dir.name)
        size_mb = Path(archive).stat().st_size / (1024 * 1024)
        print(f"  {Path(archive).name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
