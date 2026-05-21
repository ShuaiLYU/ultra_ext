"""Dataset utilities for LabelMe folders."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class LabelmeFolder:
    """A flat folder where each image has a sibling LabelMe ``.json`` file.

    Example layout::

        fabric1/images/
            foo.jpg
            foo.json
            bar.jpg
            bar.json
    """

    def __init__(self, root: str | os.PathLike, image_subdir: str = "images"):
        self.root = Path(root)
        candidate = self.root / image_subdir
        self.image_dir = candidate if candidate.is_dir() else self.root
        self.pairs: list[tuple[Path, Path | None]] = self._collect_pairs()

    def _collect_pairs(self) -> list[tuple[Path, Path | None]]:
        pairs = []
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            j = p.with_suffix(".json")
            pairs.append((p, j if j.exists() else None))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def class_counts(self) -> Counter:
        counts: Counter = Counter()
        for _, j in self.pairs:
            if j is None:
                continue
            with open(j, "r") as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                counts[shape["label"]] += 1
        return counts

    @staticmethod
    def _shape_to_xyxy(shape: dict) -> tuple[float, float, float, float] | None:
        pts = shape.get("points", [])
        st = shape.get("shape_type", "rectangle")
        if not pts:
            return None
        if st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
        elif st == "circle" and len(pts) == 2:
            (cx, cy), (ex, ey) = pts
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        elif st in ("polygon", "linestrip") and len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            return None
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def to_yolo(
        self,
        out_dir: str | os.PathLike,
        train_ratio: float = 0.9,
        seed: int = 42,
        link_mode: str = "copy",
        include_empty: bool = True,
        class_names: Iterable[str] | None = None,
    ) -> Path:
        """Convert this folder into a YOLO-format detection dataset.

        Args:
            out_dir: Output dataset root. Will contain ``images/{train,val}``,
                ``labels/{train,val}``, and ``data.yaml``.
            train_ratio: Fraction of pairs assigned to train (rest to val).
            seed: RNG seed for the shuffle.
            link_mode: ``"copy"`` (default), ``"symlink"``, or ``"hardlink"``.
            include_empty: If True, images with no shapes are kept as
                background (empty label file).
            class_names: Optional explicit class ordering. If None, classes are
                ordered by descending frequency, then alphabetically.

        Returns:
            Path to the written ``data.yaml``.
        """
        out = Path(out_dir).resolve()
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        if class_names is None:
            counts = self.class_counts()
            class_names = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        else:
            class_names = list(class_names)
        cls_to_id = {c: i for i, c in enumerate(class_names)}

        items = [(img, j) for img, j in self.pairs if include_empty or j is not None]
        rng = random.Random(seed)
        rng.shuffle(items)
        n_train = int(round(len(items) * train_ratio))
        splits = {"train": items[:n_train], "val": items[n_train:]}

        n_boxes = {"train": 0, "val": 0}
        for split, pairs in splits.items():
            for img_path, json_path in pairs:
                self._write_one(img_path, json_path, out, split, cls_to_id, link_mode, n_boxes)

        data_yaml = {
            "path": str(out),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(class_names)},
        }
        yaml_path = out / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

        print(
            f"[LabelmeFolder] {len(items)} images -> {out}\n"
            f"  train: {len(splits['train'])} imgs / {n_boxes['train']} boxes\n"
            f"  val:   {len(splits['val'])} imgs / {n_boxes['val']} boxes\n"
            f"  classes ({len(class_names)}): {class_names}\n"
            f"  data.yaml: {yaml_path}"
        )
        return yaml_path

    def _write_one(self, img_path, json_path, out, split, cls_to_id, link_mode, n_boxes):
        dst_img = out / "images" / split / img_path.name
        dst_lbl = out / "labels" / split / (img_path.stem + ".txt")
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if link_mode == "symlink":
            os.symlink(img_path.resolve(), dst_img)
        elif link_mode == "hardlink":
            os.link(img_path.resolve(), dst_img)
        elif link_mode == "copy":
            shutil.copy2(img_path, dst_img)
        else:
            raise ValueError(f"Unknown link_mode: {link_mode}")

        lines: list[str] = []
        if json_path is not None:
            with open(json_path, "r") as f:
                data = json.load(f)
            w = data.get("imageWidth")
            h = data.get("imageHeight")
            if not w or not h:
                im = cv2.imread(str(img_path))
                if im is None:
                    dst_lbl.write_text("")
                    return
                h, w = im.shape[:2]
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label not in cls_to_id:
                    continue
                xyxy = self._shape_to_xyxy(shape)
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy
                x1 = max(0.0, min(float(w), x1))
                x2 = max(0.0, min(float(w), x2))
                y1 = max(0.0, min(float(h), y1))
                y2 = max(0.0, min(float(h), y2))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cls_to_id[label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_boxes[split] += 1
        dst_lbl.write_text("\n".join(lines))
