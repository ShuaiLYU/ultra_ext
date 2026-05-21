"""Demo: convert the fabric1 LabelMe folder into a YOLO-format dataset."""

from pathlib import Path

from ultra_ext import LabelmeFolder


def main():
    base = Path("~/workspace/ultra_louis_work/buffer/AnomalyData").expanduser()
    for name in ("fabric1", "fabric2", "fabric3", "fabric4"):
        src = base / name
        dst = base / f"{name}_yolo"
        print(f"\n=== {name} ===")

        folder = LabelmeFolder(src)
        print(f"loaded {len(folder)} images from {src}")
        print(f"class counts: {dict(folder.class_counts())}")

        yaml_path = folder.to_yolo(
            out_dir=dst,
            train_ratio=0.9,
            seed=42,
            link_mode="copy",
            include_empty=True,
        )
        print(f"done -> {yaml_path}")


if __name__ == "__main__":
    main()
