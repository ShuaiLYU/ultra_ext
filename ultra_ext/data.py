"""Dataset utilities for LabelMe folders and DAGM."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _place_image(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif link_mode == "hardlink":
        os.link(src.resolve(), dst)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link_mode: {link_mode}")


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
        _place_image(img_path, dst_img, link_mode)

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


class DagmFolder:
    """The DAGM 2007 dataset root containing ``Class1`` .. ``Class10``.

    Each ``ClassX`` folder has ``Train/`` and ``Test/`` subfolders, plus a
    ``Label/`` subfolder under each holding binary masks named
    ``<stem>_label.PNG``. Images without a matching mask are normal samples.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.class_dirs: list[Path] = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and re.fullmatch(r"Class\d+", p.name)),
            key=lambda p: int(p.name.replace("Class", "")),
        )
        if not self.class_dirs:
            raise FileNotFoundError(f"No Class* directories under {self.root}")

    def __len__(self) -> int:
        return sum(len(self._list_images(c / "Train")) + len(self._list_images(c / "Test"))
                   for c in self.class_dirs)

    @staticmethod
    def _list_images(split_dir: Path) -> list[Path]:
        if not split_dir.is_dir():
            return []
        return sorted(p for p in split_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS and "label" not in p.stem.lower())

    @staticmethod
    def _mask_path(img_path: Path) -> Path:
        return img_path.parent / "Label" / f"{img_path.stem}_label{img_path.suffix}"

    def class_counts(self) -> Counter:
        """Count defect images per class (i.e. those that have a mask)."""
        counts: Counter = Counter()
        for c in self.class_dirs:
            for split in ("Train", "Test"):
                for img in self._list_images(c / split):
                    if self._mask_path(img).exists():
                        counts[c.name] += 1
        return counts

    @staticmethod
    def _mask_to_bboxes(mask_path: Path, min_area: int = 10) -> list[tuple[int, int, int, int]]:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return []
        _, binmask = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area or w < 2 or h < 2:
                continue
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    def to_yolo(
        self,
        out_dir: str | os.PathLike,
        split: str = "official",
        train_ratio: float = 0.9,
        seed: int = 42,
        link_mode: str = "copy",
        include_normal: bool = True,
        min_area: int = 10,
    ) -> Path:
        """Convert DAGM into a YOLO-format detection dataset.

        Args:
            out_dir: Output dataset root.
            split: ``"official"`` (default) uses each class's ``Train``→train
                and ``Test``→val. ``"random"`` pools everything and re-splits
                by ``train_ratio``.
            train_ratio: Only used when ``split="random"``.
            seed: RNG seed for the random split.
            link_mode: ``"copy"`` (default), ``"symlink"``, or ``"hardlink"``.
            include_normal: Keep images without a mask as background.
            min_area: Skip mask connected components smaller than this.

        Returns:
            Path to the written ``data.yaml``.
        """
        out = Path(out_dir).resolve()
        for sp in ("train", "val"):
            (out / "images" / sp).mkdir(parents=True, exist_ok=True)
            (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

        class_names = [c.name for c in self.class_dirs]
        cls_to_id = {c: i for i, c in enumerate(class_names)}

        # Gather (img_path, cls_name, official_split) tuples
        items: list[tuple[Path, str, str]] = []
        for c in self.class_dirs:
            for sp in ("Train", "Test"):
                for img in self._list_images(c / sp):
                    if not include_normal and not self._mask_path(img).exists():
                        continue
                    items.append((img, c.name, "train" if sp == "Train" else "val"))

        if split == "official":
            assignments = [(img, cls, sp) for img, cls, sp in items]
        elif split == "random":
            rng = random.Random(seed)
            rng.shuffle(items)
            n_train = int(round(len(items) * train_ratio))
            assignments = [
                (img, cls, "train" if i < n_train else "val")
                for i, (img, cls, _) in enumerate(items)
            ]
        else:
            raise ValueError(f"Unknown split: {split}")

        n_imgs = {"train": 0, "val": 0}
        n_boxes = {"train": 0, "val": 0}
        n_bg = {"train": 0, "val": 0}
        # Image stems may collide across classes (e.g. 0001.PNG appears in
        # every class). Prefix with class name to disambiguate.
        for img_path, cls_name, sp in assignments:
            stem = f"{cls_name}_{img_path.stem}"
            dst_img = out / "images" / sp / f"{stem}{img_path.suffix}"
            dst_lbl = out / "labels" / sp / f"{stem}.txt"
            _place_image(img_path, dst_img, link_mode)

            mask_path = self._mask_path(img_path)
            lines: list[str] = []
            if mask_path.exists():
                im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if im is None:
                    dst_lbl.write_text("")
                    continue
                h, w = im.shape[:2]
                for x1, y1, x2, y2 in self._mask_to_bboxes(mask_path, min_area=min_area):
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    nw = (x2 - x1) / w
                    nh = (y2 - y1) / h
                    lines.append(f"{cls_to_id[cls_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    n_boxes[sp] += 1
            else:
                n_bg[sp] += 1
            dst_lbl.write_text("\n".join(lines))
            n_imgs[sp] += 1

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
            f"[DagmFolder] {sum(n_imgs.values())} images -> {out}\n"
            f"  train: {n_imgs['train']} imgs ({n_bg['train']} bg) / {n_boxes['train']} boxes\n"
            f"  val:   {n_imgs['val']} imgs ({n_bg['val']} bg) / {n_boxes['val']} boxes\n"
            f"  classes ({len(class_names)}): {class_names}\n"
            f"  data.yaml: {yaml_path}"
        )
        return yaml_path

    def to_yolo_per_class(
        self,
        out_dir: str | os.PathLike,
        train_ratio: float = 0.9,
        seed: int = 42,
        link_mode: str = "copy",
        include_normal: bool = True,
        min_area: int = 10,
        class_name: str | None = None,
    ) -> list[Path]:
        """Convert each ``ClassN`` into its own single-class YOLO dataset.

        Produces ``<out_dir>/ClassN/`` with its own ``images/{train,val}``,
        ``labels/{train,val}``, and ``data.yaml``. The split is random within
        each class so train/val ratio is exact per-class.

        Args:
            class_name: Override the class name written to ``data.yaml``
                (e.g. ``"anomaly"``). When None, uses the source folder name.

        Returns:
            List of paths to the per-class ``data.yaml`` files.
        """
        out_root = Path(out_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        yaml_paths: list[Path] = []

        for c in self.class_dirs:
            cls_name = c.name
            sub_out = out_root / cls_name
            for sp in ("train", "val"):
                (sub_out / "images" / sp).mkdir(parents=True, exist_ok=True)
                (sub_out / "labels" / sp).mkdir(parents=True, exist_ok=True)

            imgs: list[Path] = []
            for sp in ("Train", "Test"):
                for img in self._list_images(c / sp):
                    if not include_normal and not self._mask_path(img).exists():
                        continue
                    imgs.append(img)

            rng = random.Random(seed)
            rng.shuffle(imgs)
            n_train = int(round(len(imgs) * train_ratio))
            assignments = [(p, "train" if i < n_train else "val") for i, p in enumerate(imgs)]

            n_imgs = {"train": 0, "val": 0}
            n_boxes = {"train": 0, "val": 0}
            n_bg = {"train": 0, "val": 0}
            for img_path, sp in assignments:
                # Stems may repeat across Train/Test (e.g. 0001.PNG in both Train and Test).
                # Prefix with the official split to disambiguate.
                src_split_tag = img_path.parent.name  # "Train" or "Test"
                stem = f"{src_split_tag}_{img_path.stem}"
                dst_img = sub_out / "images" / sp / f"{stem}{img_path.suffix}"
                dst_lbl = sub_out / "labels" / sp / f"{stem}.txt"
                _place_image(img_path, dst_img, link_mode)

                mask_path = self._mask_path(img_path)
                lines: list[str] = []
                if mask_path.exists():
                    im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if im is None:
                        dst_lbl.write_text("")
                        continue
                    h, w = im.shape[:2]
                    for x1, y1, x2, y2 in self._mask_to_bboxes(mask_path, min_area=min_area):
                        cx = (x1 + x2) / 2.0 / w
                        cy = (y1 + y2) / 2.0 / h
                        nw = (x2 - x1) / w
                        nh = (y2 - y1) / h
                        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        n_boxes[sp] += 1
                else:
                    n_bg[sp] += 1
                dst_lbl.write_text("\n".join(lines))
                n_imgs[sp] += 1

            data_yaml = {
                "path": str(sub_out),
                "train": "images/train",
                "val": "images/val",
                "names": {0: class_name or cls_name},
            }
            yaml_path = sub_out / "data.yaml"
            with open(yaml_path, "w") as f:
                yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
            yaml_paths.append(yaml_path)

            ratio = n_imgs["val"] / max(1, sum(n_imgs.values()))
            print(
                f"[DagmFolder/{cls_name}] {sum(n_imgs.values())} imgs -> {sub_out} "
                f"| train {n_imgs['train']} ({n_bg['train']} bg)/{n_boxes['train']} boxes "
                f"| val {n_imgs['val']} ({n_bg['val']} bg)/{n_boxes['val']} boxes "
                f"| val_ratio={ratio:.3f}"
            )

        return yaml_paths
