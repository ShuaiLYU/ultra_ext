"""Demo: convert DAGM (Class1..Class10) into MVTec-AD format, one per class."""

import shutil
from pathlib import Path

from ultra_ext import DagmFolder


def main():
    src = Path("~/workspace/ultra_louis_work/buffer/AnomalyData/DAGM").expanduser()
    dst = Path("~/workspace/ultra_louis_work/buffer/AnomalyData/DAGM_AD").expanduser()

    folder = DagmFolder(src)
    print(f"loaded {len(folder)} images from {src}")
    print(f"defect counts (per class): {dict(folder.class_counts())}\n")

    cls_dirs = folder.to_mvtec_ad_per_class(
        out_dir=dst,
        train_ratio=0.9,
        seed=42,
        link_mode="copy",
        defect_name="defect",
    )
    print(f"\ndone -> {len(cls_dirs)} per-class MVTec-AD datasets under {dst}\n")

    print("zipping per-class datasets ...")
    for cls_dir in cls_dirs:
        zip_base = dst / f"DAGM_AD_{cls_dir.name}"
        archive = shutil.make_archive(str(zip_base), "zip", root_dir=dst, base_dir=cls_dir.name)
        size_mb = Path(archive).stat().st_size / (1024 * 1024)
        print(f"  {Path(archive).name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
