"""
Objects365v1 dataset index reader & visualizer.
Usage:
    ds = Objects365Dataset("../buffer/object365v1_cls2imgs.csv")
    samples = ds.get_samples_by_cls(cls_id=3, max_samples=20)
    ds.visualize(samples, cls_id=3, cols=4, show=True, save_dir="./buffer/vis")
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

NUM_CLASSES = 365

CLS_NAMES: dict[int, str] = {
    0: "person", 1: "sneakers", 2: "chair", 3: "hat", 4: "lamp",
    5: "bottle", 6: "cabinet/shelf", 7: "cup", 8: "car", 9: "glasses",
    10: "picture/frame", 11: "desk", 12: "handbag", 13: "street lights", 14: "book",
    15: "plate", 16: "helmet", 17: "leather shoes", 18: "pillow", 19: "glove",
    20: "potted plant", 21: "bracelet", 22: "flower", 23: "tv", 24: "storage box",
    25: "vase", 26: "bench", 27: "wine glass", 28: "boots", 29: "bowl",
    30: "dining table", 31: "umbrella", 32: "boat", 33: "flag", 34: "speaker",
    35: "trash bin/can", 36: "stool", 37: "backpack", 38: "couch", 39: "belt",
    40: "carpet", 41: "basket", 42: "towel/napkin", 43: "slippers", 44: "barrel/bucket",
    45: "coffee table", 46: "suv", 47: "toy", 48: "tie", 49: "bed",
    50: "traffic light", 51: "pen/pencil", 52: "microphone", 53: "sandals", 54: "canned",
    55: "necklace", 56: "mirror", 57: "faucet", 58: "bicycle", 59: "bread",
    60: "high heels", 61: "ring", 62: "van", 63: "watch", 64: "sink",
    65: "horse", 66: "fish", 67: "apple", 68: "camera", 69: "candle",
    70: "teddy bear", 71: "cake", 72: "motorcycle", 73: "wild bird", 74: "laptop",
    75: "knife", 76: "traffic sign", 77: "cell phone", 78: "paddle", 79: "truck",
    80: "cow", 81: "power outlet", 82: "clock", 83: "drum", 84: "fork",
    85: "bus", 86: "hanger", 87: "nightstand", 88: "pot/pan", 89: "sheep",
    90: "guitar", 91: "traffic cone", 92: "tea pot", 93: "keyboard", 94: "tripod",
    95: "hockey", 96: "fan", 97: "dog", 98: "spoon", 99: "blackboard/whiteboard",
    100: "balloon", 101: "air conditioner", 102: "cymbal", 103: "mouse", 104: "telephone",
    105: "pickup truck", 106: "orange", 107: "banana", 108: "airplane", 109: "luggage",
    110: "skis", 111: "soccer", 112: "trolley", 113: "oven", 114: "remote",
    115: "baseball glove", 116: "paper towel", 117: "refrigerator", 118: "train", 119: "tomato",
    120: "machinery vehicle", 121: "tent", 122: "shampoo/shower gel", 123: "head phone", 124: "lantern",
    125: "donut", 126: "cleaning products", 127: "sailboat", 128: "tangerine", 129: "pizza",
    130: "kite", 131: "computer box", 132: "elephant", 133: "toiletries", 134: "gas stove",
    135: "broccoli", 136: "toilet", 137: "stroller", 138: "shovel", 139: "baseball bat",
    140: "microwave", 141: "skateboard", 142: "surfboard", 143: "surveillance camera", 144: "gun",
    145: "life saver", 146: "cat", 147: "lemon", 148: "liquid soap", 149: "zebra",
    150: "duck", 151: "sports car", 152: "giraffe", 153: "pumpkin", 154: "piano",
    155: "stop sign", 156: "radiator", 157: "converter", 158: "tissue", 159: "carrot",
    160: "washing machine", 161: "vent", 162: "cookies", 163: "cutting/chopping board", 164: "tennis racket",
    165: "candy", 166: "skating and skiing shoes", 167: "scissors", 168: "folder", 169: "baseball",
    170: "strawberry", 171: "bow tie", 172: "pigeon", 173: "pepper", 174: "coffee machine",
    175: "bathtub", 176: "snowboard", 177: "suitcase", 178: "grapes", 179: "ladder",
    180: "pear", 181: "american football", 182: "basketball", 183: "potato", 184: "paint brush",
    185: "printer", 186: "billiards", 187: "fire hydrant", 188: "goose", 189: "projector",
    190: "sausage", 191: "fire extinguisher", 192: "extension cord", 193: "facial mask", 194: "tennis ball",
    195: "chopsticks", 196: "electronic stove and gas stove", 197: "pie", 198: "frisbee", 199: "kettle",
    200: "hamburger", 201: "golf club", 202: "cucumber", 203: "clutch", 204: "blender",
    205: "tong", 206: "slide", 207: "hot dog", 208: "toothbrush", 209: "facial cleanser",
    210: "mango", 211: "deer", 212: "egg", 213: "violin", 214: "marker",
    215: "ship", 216: "chicken", 217: "onion", 218: "ice cream", 219: "tape",
    220: "wheelchair", 221: "plum", 222: "bar soap", 223: "scale", 224: "watermelon",
    225: "cabbage", 226: "router/modem", 227: "golf ball", 228: "pine apple", 229: "crane",
    230: "fire truck", 231: "peach", 232: "cello", 233: "notepaper", 234: "tricycle",
    235: "toaster", 236: "helicopter", 237: "green beans", 238: "brush", 239: "carriage",
    240: "cigar", 241: "earphone", 242: "penguin", 243: "hurdle", 244: "swing",
    245: "radio", 246: "cd", 247: "parking meter", 248: "swan", 249: "garlic",
    250: "french fries", 251: "horn", 252: "avocado", 253: "saxophone", 254: "trumpet",
    255: "sandwich", 256: "cue", 257: "kiwi fruit", 258: "bear", 259: "fishing rod",
    260: "cherry", 261: "tablet", 262: "green vegetables", 263: "nuts", 264: "corn",
    265: "key", 266: "screwdriver", 267: "globe", 268: "broom", 269: "pliers",
    270: "volleyball", 271: "hammer", 272: "eggplant", 273: "trophy", 274: "dates",
    275: "board eraser", 276: "rice", 277: "tape measure/ruler", 278: "dumbbell", 279: "hamimelon",
    280: "stapler", 281: "camel", 282: "lettuce", 283: "goldfish", 284: "meat balls",
    285: "medal", 286: "toothpaste", 287: "antelope", 288: "shrimp", 289: "rickshaw",
    290: "trombone", 291: "pomegranate", 292: "coconut", 293: "jellyfish", 294: "mushroom",
    295: "calculator", 296: "treadmill", 297: "butterfly", 298: "egg tart", 299: "cheese",
    300: "pig", 301: "pomelo", 302: "race car", 303: "rice cooker", 304: "tuba",
    305: "crosswalk sign", 306: "papaya", 307: "hair drier", 308: "green onion", 309: "chips",
    310: "dolphin", 311: "sushi", 312: "urinal", 313: "donkey", 314: "electric drill",
    315: "spring rolls", 316: "tortoise/turtle", 317: "parrot", 318: "flute", 319: "measuring cup",
    320: "shark", 321: "steak", 322: "poker card", 323: "binoculars", 324: "llama",
    325: "radish", 326: "noodles", 327: "yak", 328: "mop", 329: "crab",
    330: "microscope", 331: "barbell", 332: "bread/bun", 333: "baozi", 334: "lion",
    335: "red cabbage", 336: "polar bear", 337: "lighter", 338: "seal", 339: "mangosteen",
    340: "comb", 341: "eraser", 342: "pitaya", 343: "scallop", 344: "pencil case",
    345: "saw", 346: "table tennis paddle", 347: "okra", 348: "starfish", 349: "eagle",
    350: "monkey", 351: "durian", 352: "game board", 353: "rabbit", 354: "french horn",
    355: "ambulance", 356: "asparagus", 357: "hoverboard", 358: "pasta", 359: "target",
    360: "hotair balloon", 361: "chainsaw", 362: "lobster", 363: "iron", 364: "flashlight",
}


@dataclass
class Sample:
    img_path: Path
    label_path: Path


class Object365V1Dataset:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        assert self.csv_path.exists(), f"CSV not found: {self.csv_path}"

        self._cls_cols = [f"cls_{i:03d}" for i in range(NUM_CLASSES)]
        print(f"[info] loading {self.csv_path} …")
        self.df = pd.read_csv(
            self.csv_path,
            dtype={c: np.int32 for c in self._cls_cols},
        )
        self.cls_names: dict[int, str] = CLS_NAMES
        self.name2cls: dict[str, int] = {v: k for k, v in CLS_NAMES.items()}
        print(f"[info] loaded {len(self.df):,} images")

    # ── public API ────────────────────────────────────────────────────────────

    def get_samples_by_cls(
        self,
        cls_id: int | str,
        max_samples: int | None = None,
        shuffle: bool = False,
    ) -> list[Sample]:
        """
        Return images that contain at least one bbox of `cls_id`.

        Args:
            cls_id:      class index (0–364) or class name string
            max_samples: cap the result list (None = all)
            shuffle:     randomly shuffle before capping
        """
        cls_id = self._resolve_cls(cls_id)
        col = f"cls_{cls_id:03d}"
        mask = self.df[col] > 0
        sub = self.df[mask]

        if shuffle:
            sub = sub.sample(frac=1, random_state=42)
        if max_samples is not None:
            sub = sub.iloc[:max_samples]

        samples = [
            Sample(img_path=Path(r.img_path), label_path=Path(r.label_path))
            for r in sub.itertuples()
        ]
        print(f"[info] cls {cls_id} ({self.cls_names[cls_id]}): {mask.sum():,} total images, returning {len(samples)}")
        return samples

    def debug_label(self, sample: Sample, cls_id: int | str) -> None:
        """Print raw label lines and computed pixel coords for diagnosis."""
        cls_id = self._resolve_cls(cls_id)
        img = self._load_image(sample.img_path)
        if img is None:
            print(f"[error] cannot load image: {sample.img_path}")
            return
        h, w = img.shape[:2]
        print(f"image : {sample.img_path.name}  size=({w}w x {h}h)")
        print(f"label : {sample.label_path}")
        print(f"{'raw line (truncated)':<60s}  ->  x1     y1     x2     y2")
        with open(sample.label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or int(parts[0]) != cls_id:
                    continue
                coords = list(map(float, parts[1:]))
                xs = [coords[i] * w for i in range(0, len(coords), 2)]
                ys = [coords[i] * h for i in range(1, len(coords), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                preview = line.strip()[:58]
                print(f"  {preview:<58s}  ->  {x1:6.1f} {y1:6.1f} {x2:6.1f} {y2:6.1f}")


    def export_cls_images(self,cls_id,save_dir,draw_bbox=True,box_color=(0,0,255),line_width=2,max_samples=None):
        """
        Iterate all images containing `cls_id` and save them (with optional
        bbox overlay) to `save_dir/[imgname].jpg`.
        """
        cls_id = self._resolve_cls(cls_id)
        cls_name = self.cls_names[cls_id]
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        samples = self.get_samples_by_cls(cls_id, max_samples=max_samples)


        print(f"[info] exporting {len(samples):,} images for cls {cls_id} ({cls_name}) → {save_dir}")

        skipped = 0
        for sample in tqdm(samples, desc=f"Exporting '{cls_name}'", unit="img"):
            if not sample.img_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(sample.img_path))
            if img is None:
                skipped += 1
                continue

            if draw_bbox:
                h, w = img.shape[:2]
                for x1, y1, x2, y2 in self._load_bboxes_for_cls(sample.label_path, cls_id, w, h):
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, line_width)

            # out_path = save_dir / (sample.img_path.stem + ".jpg")
            out_path=save_dir / sample.img_path.name
            cv2.imwrite(str(out_path), img)

        print(f"[done] saved {len(samples) - skipped:,} images  (skipped {skipped})")


    def export_cls_images_predict(
        self,
        cls_id: int | str,
        save_dir: str | Path,
        draw_bbox: bool = True,
        box_color: tuple[int, int, int] = (0, 0, 255),  # BGR
        line_width: int = 2,
        yoloe_model: str | None = None,
        yoloe_names: list[str] | None = None,
        yoloe_conf: float = 0.1,
        max_samples=None,
        mode="text"
    ) -> None:
        """
        Iterate all images containing `cls_id` and save them (with optional
        bbox overlay) to `save_dir/[imgname].jpg`.
        If `yoloe_model` is provided, the saved image is a side-by-side
        composite: GT boxes on the left, YOLOE prediction on the right.

        Args:
            cls_id:       class index or name string
            save_dir:     output directory
            draw_bbox:    whether to draw GT bboxes on the left panel
            box_color:    BGR color for drawn boxes
            line_width:   box line thickness in pixels
            yoloe_model:  path to YOLOE weights (e.g. "yoloe-26l-seg.pt")
            yoloe_names:  class name list passed to YOLOE; defaults to [cls_name]
            yoloe_conf:   confidence threshold for YOLOE prediction
            mode:        "text" (default) or "visual"
        """

        assert mode in ("text", "visual"), f"invalid mode: {mode}"


        cls_id = self._resolve_cls(cls_id)
        cls_name = self.cls_names[cls_id]
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        samples = self.get_samples_by_cls(cls_id, max_samples=max_samples)
        print(f"[info] exporting {len(samples):,} images for cls {cls_id} ({cls_name}) → {save_dir}")

        # load YOLOE model once before the loop
        _yoloe = None
        if yoloe_model is not None:
            from ultralytics import YOLO,YOLOE

            # _yoloe = YOLO(yoloe_model)   
            _yoloe = YOLOE(yoloe_model)  

            if mode == "text":
                _names = yoloe_names or [cls_name]
                _yoloe.set_classes(_names, _yoloe.get_text_pe(_names))
                print(f"[info] YOLOE model loaded: {yoloe_model}, names={_names}")
   

        skipped = 0
        for sample in tqdm(samples, desc=f"Exporting '{cls_name}'", unit="img"):
            if not sample.img_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(sample.img_path))
            if img is None:
                skipped += 1
                continue

            if draw_bbox:
                h, w = img.shape[:2]
                for x1, y1, x2, y2 in self._load_bboxes_for_cls(sample.label_path, cls_id, w, h):
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, line_width)

            out_path = save_dir / (sample.img_path.stem + ".jpg")

            if _yoloe is not None:
                # res.plot() returns a BGR numpy array with detections drawn
                if mode == "text":
                    res = _yoloe.predict(source=str(sample.img_path), conf=yoloe_conf, verbose=False)[0]
                    pred_img = res.plot()
                else:
                    bboxes=self._load_bboxes_for_cls(sample.label_path, cls_id, w, h)
                    visual_prompts = dict(
                        bboxes=np.array(bboxes[0])[None],
                        cls=np.array([0]),
                    )

                    res = _yoloe.predict(source=str(sample.img_path), conf=yoloe_conf, 
                                           visual_prompts=visual_prompts,verbose=False)[0]

                    pred_img = res.plot()
                    

                    # draw visual_prompts 
                    for cls,bbox in zip(visual_prompts["cls"],visual_prompts["bboxes"]):
                        x1,y1,x2,y2=bbox.tolist()
                        cv2.rectangle(pred_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), line_width)
                        cv2.putText(pred_img, f"prompt_cls{cls}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


                # resize prediction panel to match GT panel height
                gt_h, gt_w = img.shape[:2]
                pred_h, pred_w = pred_img.shape[:2]
                if pred_h != gt_h:
                    scale = gt_h / pred_h
                    pred_img = cv2.resize(pred_img, (int(pred_w * scale), gt_h))

                # combined = np.hstack([img, pred_img])
                cv2.imwrite(str(out_path), pred_img)
            else:
                cv2.imwrite(str(out_path), img)

        print(f"[done] saved {len(samples) - skipped:,} images  (skipped {skipped})")

    def visualize(
        self,
        sample: Sample,
        cls_id: int | str,
        box_color: tuple = (1.0, 0.2, 0.2),
        line_width: float = 2.0,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> None:
        cls_id = self._resolve_cls(cls_id)

        img = self._load_image(sample.img_path)
        if img is None:
            print(f"[warn] cannot load image: {sample.img_path}")
            return

        h, w = img.shape[:2]
        bboxes = self._load_bboxes_for_cls(sample.label_path, cls_id, w, h)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.axis("off")

        for x1, y1, x2, y2 in bboxes:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=line_width,
                edgecolor=box_color,
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(
            f"cls {cls_id} · {self.cls_names[cls_id]} · {len(bboxes)} bbox(es) · {sample.img_path.name}",
            fontsize=9,
        )
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"[info] saved → {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def export_stats_csv(self, save_path: str | Path) -> None:
        """Save per-class statistics to a CSV file.

        Columns: cls, name, boxes (total bbox count), images (image count).
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for cls_id in range(NUM_CLASSES):
            col = f"cls_{cls_id:03d}"
            box_count = int(self.df[col].sum())
            img_count = int((self.df[col] > 0).sum())
            rows.append({
                "cls":    cls_id,
                "name":   self.cls_names[cls_id],
                "boxes":  box_count,
                "images": img_count,
            })
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"[info] stats saved → {save_path}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_cls(self, cls_id: int) -> None:
        assert 0 <= cls_id < NUM_CLASSES, f"cls_id must be 0–{NUM_CLASSES - 1}, got {cls_id}"

    def _resolve_cls(self, cls_id: int | str) -> int:
        """Accept either an int index or a class name string."""
        if isinstance(cls_id, str):
            assert cls_id in self.name2cls, f"Unknown class name: '{cls_id}'"
            cls_id = self.name2cls[cls_id]
        self._check_cls(cls_id)
        return cls_id

    @staticmethod
    def _load_image(img_path: Path) -> np.ndarray | None:
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_bboxes_for_cls(
        label_path: Path,
        cls_id: int,
        img_w: int,
        img_h: int,
    ) -> list[tuple[float, float, float, float]]:
        """
        Parse YOLO segmentation format: cls_id x1 y1 x2 y2 ... xn yn (normalised).
        Returns pixel-space bounding boxes (x1, y1, x2, y2) derived from the polygon.
        """
        bboxes = []
        if not label_path.exists():
            return bboxes
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or int(parts[0]) != cls_id:
                    continue
                coords = list(map(float, parts[1:]))
                if len(coords) < 2:
                    continue
                xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
                ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        return bboxes





class Object365V1DatasetCache(Object365V1Dataset):
    """
    Read an Ultralytics-format `.cache` file and provide the same API
    as Object365V1Dataset (inherits export_cls_images, export_cls_images_predict,
    visualize, debug_label, _load_image, _load_bboxes_for_cls, _resolve_cls, etc.).

    Detection style  (no "texts" key):
        cls  – global category index (0–364), directly from CLS_NAMES.

    Grounding style  ("texts" key present per label):
        texts – [['cat'], ['dog', 'puppy'], ...]   per-image category groups
        cls   – LOCAL index into that image's texts list.

        On load, all unique canonical texts (first element of each group) are
        collected into a global vocabulary and each label's cls is remapped
        from local → global index.  After init, both styles expose identical
        cls_names / name2cls / _cls_index structures.
    """

    def __init__(self, cache_path: str | Path):
        # intentionally skip Object365V1Dataset.__init__ (needs a CSV)
        self.cache_path = Path(cache_path)
        assert self.cache_path.exists(), f"Cache not found: {self.cache_path}"

        print(f"[info] loading {self.cache_path} …")
        raw = np.load(str(self.cache_path), allow_pickle=True).item()

        self.labels: list[dict] = raw["labels"]
        self.version = raw.get("version", None)
        self.hash    = raw.get("hash", None)
        self.msgs    = raw.get("msgs", [])
        self.results = raw.get("results", None)   # (nf, nm, ne, nc, total)

        # ── detect style ──────────────────────────────────────────────────────
        self.data_style = "detection"
        for label in self.labels:
            if label.get("texts") is not None:
                self.data_style = "grounding"
                break
        print(f"[info] data_style='{self.data_style}'")

        # ── build global vocab & _cls_index ───────────────────────────────────
        print("[info] building global index …")
        if self.data_style == "detection":
            self.cls_names: dict[int, str] = CLS_NAMES
            self.name2cls:  dict[str, int] = {v: k for k, v in CLS_NAMES.items()}
            self._num_classes = NUM_CLASSES
            self._cls_index: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
            for idx, label in enumerate(self.labels):
                cls_arr = label.get("cls")
                if cls_arr is None or len(cls_arr) == 0:
                    continue
                for g in np.unique(cls_arr.astype(int).ravel()):
                    self._cls_index[int(g)].append(idx)

        else:  # grounding
            # Pass 1: collect global vocab (canonical = first text in each group)
            global_texts: list[str] = []
            text2gidx: dict[str, int] = {}
            for label in self.labels:
                for group in (label.get("texts") or []):
                    if not group:
                        continue
                    canon = group[0]
                    if canon not in text2gidx:
                        text2gidx[canon] = len(global_texts)
                        global_texts.append(canon)

            self.cls_names  = {i: t for i, t in enumerate(global_texts)}
            self.name2cls   = text2gidx
            self._num_classes = len(global_texts)
            self._cls_index = {i: [] for i in range(self._num_classes)}

            # Pass 2: remap each label's cls from local → global and build index
            for idx, label in enumerate(self.labels):
                texts   = label.get("texts")
                cls_arr = label.get("cls")
                if not texts or cls_arr is None or len(cls_arr) == 0:
                    continue
                local2global = [
                    text2gidx[group[0]]
                    for group in texts
                    if group
                ]
                new_cls = [
                    local2global[int(c)]
                    for c in cls_arr.ravel()
                    if int(c) < len(local2global)
                ]
                label["cls"] = np.array(new_cls, dtype=np.int32).reshape(-1, 1)
                for g in set(new_cls):
                    self._cls_index[g].append(idx)

        print(f"[info] loaded {len(self.labels):,} images, {self._num_classes:,} classes")

    # ── overrides ─────────────────────────────────────────────────────────────

    def _get_label_by_imfile(self, im_file: str | Path) -> dict | None:
        """Fast lookup of a label dict by im_file path."""
        if not hasattr(self, "_imfile2label"):
            self._imfile2label = {label["im_file"]: label for label in self.labels}
        return self._imfile2label.get(str(im_file))

    def _load_bboxes_from_cache(
        self,
        im_file: str | Path,
        cls_id: int,
        img_w: int,
        img_h: int,
    ) -> list[tuple[float, float, float, float]]:
        """
        Return pixel-space (x1,y1,x2,y2) bboxes for `cls_id` from the in-memory cache.
        Bboxes in cache are normalized xywh; converts to pixel xyxy.
        """
        label = self._get_label_by_imfile(im_file)
        if label is None:
            return []
        cls_arr  = label.get("cls")
        bbox_arr = label.get("bboxes")
        if cls_arr is None or bbox_arr is None or len(cls_arr) == 0:
            return []

        bboxes = []
        for c, box in zip(cls_arr.ravel(), bbox_arr):
            if int(c) != cls_id:
                continue
            cx, cy, w, h = box
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            bboxes.append((x1, y1, x2, y2))
        return bboxes

    def export_cls_images(
        self,
        cls_id,
        save_dir,
        draw_bbox=True,
        box_color=(0, 0, 255),
        line_width=2,
        max_samples=None,
    ):
        """Export images with bboxes drawn from the in-memory cache."""
        cls_id   = self._resolve_cls(cls_id)
        cls_name = self.cls_names[cls_id]
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        samples = self.get_samples_by_cls(cls_id, max_samples=max_samples)
        print(f"[info] exporting {len(samples):,} images for cls {cls_id} ({cls_name}) → {save_dir}")

        skipped = 0
        for sample in tqdm(samples, desc=f"Exporting '{cls_name}'", unit="img"):
            if not sample.img_path.exists():
                skipped += 1
                continue
            img = cv2.imread(str(sample.img_path))
            if img is None:
                skipped += 1
                continue

            if draw_bbox:
                h, w = img.shape[:2]
                for x1, y1, x2, y2 in self._load_bboxes_from_cache(sample.img_path, cls_id, w, h):
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, line_width)

            out_path = save_dir / sample.img_path.name
            cv2.imwrite(str(out_path), img)

        print(f"[done] saved {len(samples) - skipped:,} images  (skipped {skipped})")

    def _check_cls(self, cls_id: int) -> None:
        assert 0 <= cls_id < self._num_classes, (
            f"cls_id must be 0–{self._num_classes - 1}, got {cls_id}"
        )

    def get_samples_by_cls(
        self,
        cls_id: int | str,
        max_samples: int | None = None,
        shuffle: bool = False,
    ) -> list[Sample]:
        """
        Return Sample objects for images containing at least one bbox of `cls_id`.

        Args:
            cls_id:      global class index or class/text name string
            max_samples: cap the result list (None = all)
            shuffle:     randomly shuffle before capping
        """
        cls_id  = self._resolve_cls(cls_id)
        indices = list(self._cls_index[cls_id])

        if shuffle:
            import random
            random.shuffle(indices)
        if max_samples is not None:
            indices = indices[:max_samples]

        samples = [
            Sample(
                img_path=Path(self.labels[i]["im_file"]),
                label_path=Path(self.labels[i]["im_file"]).with_suffix(".txt"),
            )
            for i in indices
        ]
        print(f"[info] cls {cls_id} ({self.cls_names[cls_id]}): "
              f"{len(self._cls_index[cls_id]):,} total, returning {len(samples)}")
        return samples

    # ── additional helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def get_label(self, idx: int) -> dict:
        """Return the raw label dict at position `idx`."""
        return self.labels[idx]

    def cls_counts(self) -> dict[int, int]:
        """Return {cls_id: image_count} for all classes."""
        return {k: len(v) for k, v in self._cls_index.items()}

    def print_stats(self) -> None:
        """Print per-class image counts."""
        counts = self.cls_counts()
        print(f"Cache : {self.cache_path}")
        print(f"Style : {self.data_style}")
        print(f"Images: {len(self.labels):,}  |  Classes: {self._num_classes:,}")
        if self.results:
            nf, nm, ne, nc, n = self.results
            print(f"Found/missing/empty/corrupt: {nf}/{nm}/{ne}/{nc} of {n}")
        print(f"\n  {'cls':>6}  {'name':<50s}  {'images':>7}")
        for cls_id, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt == 0:
                continue
            print(f"  {cls_id:>6}  {self.cls_names[cls_id]:<50s}  {cnt:>7,}")

    def export_stats_csv(self, save_path: str | Path) -> None:
        """Save per-class statistics to a CSV file.

        Columns: cls, name, boxes (total bbox count), images (image count).
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # count total boxes per class by iterating all labels
        box_counts: dict[int, int] = {i: 0 for i in range(self._num_classes)}
        for label in self.labels:
            cls_arr = label.get("cls")
            if cls_arr is None or len(cls_arr) == 0:
                continue
            for c in cls_arr.ravel():
                box_counts[int(c)] += 1

        rows = [
            {
                "cls":    cls_id,
                "name":   self.cls_names[cls_id],
                "boxes":  box_counts[cls_id],
                "images": len(self._cls_index[cls_id]),
            }
            for cls_id in range(self._num_classes)
        ]
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"[info] stats saved → {save_path}")


if __name__ == "__main__":
    # # Example usage:
    # cache = Object365V1DatasetCache("../datasets/Objects365v1/annotations/objects365_train_segm.engine.cache")

    # cache.export_stats_csv("../buffer/objects365v1_de_stats.csv")
    # # samples = cache.get_samples_by_cls("hat", max_samples=5, shuffle=True)
    # # for sample in samples:
    # #     print(sample)


    target_cls_list= ['lobster','pasta','chainsaw','iron','asparagus','monkey']
    target_cls_list+= ['person','stop sign',"dog"]

    ds = Object365V1DatasetCache("../datasets/Objects365v1/annotations/objects365_train_segm.engine.cache")
    for TARGET_CLS in target_cls_list:
        ds.export_cls_images(
            TARGET_CLS,
            save_dir="../buffer/temp/obj365v1_{}_anno_engine".format(TARGET_CLS),
            draw_bbox=True,
            box_color=(255, 0, 0),
            line_width=3,
            max_samples=30
            )
    