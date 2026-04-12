
def load_labels_from_cache( cache_file):
    from ultralytics.data.utils import load_dataset_cache_file
    cache=load_dataset_cache_file(cache_file)
    labels=cache['labels']
    return labels






def print_first_n_labels_from_cache(cache_file, n=3):
    from ultra_ext.utils import super_print
    labels=load_labels_from_cache(cache_file)
    print(f"First {n} labels from cache {cache_file}:")
    for i in range(min(n, len(labels))):
        super_print(f"Label {i}", labels[i])





"""

First 2 labels from cache ../datasets/Objects365v1/annotations/objects365_train_segm.engine.segment.cache:
Label 0: dict with keys ['im_file', 'shape', 'texts', 'bboxes', 'cls', 'normalized', 'bbox_format', 'segments']
Label 0['im_file']: ../datasets/Objects365v1/images/train/obj365_train_000000410648.jpg (str)
Label 0['shape']: list of length 2
Label 0['shape'] [0]: 512 (int)
Label 0['texts']: list of length 6
Label 0['texts'] [0]: list of length 1
Label 0['texts'] [0] [0]: umbrella (str)
Label 0['bboxes']: shape=(16, 4), dtype=float32
Label 0['cls']: shape=(16, 1), dtype=float32
Label 0['normalized']: True (bool)
Label 0['bbox_format']: xywh (str)
Label 0['segments']: list of length 16
Label 0['segments'] [0]: shape=(7, 2), dtype=float32
Label 1: dict with keys ['im_file', 'shape', 'texts', 'bboxes', 'cls', 'normalized', 'bbox_format', 'segments']
Label 1['im_file']: ../datasets/Objects365v1/images/train/obj365_train_000000467051.jpg (str)
Label 1['shape']: list of length 2
Label 1['shape'] [0]: 768 (int)
Label 1['texts']: list of length 8
Label 1['texts'] [0]: list of length 1
Label 1['texts'] [0] [0]: bracelet (str)
Label 1['bboxes']: shape=(18, 4), dtype=float32
Label 1['cls']: shape=(18, 1), dtype=float32
Label 1['normalized']: True (bool)
Label 1['bbox_format']: xywh (str)
Label 1['segments']: list of length 18
Label 1['segments'] [0]: shape=(20, 2), dtype=float32

"""



class UltraCache():
    def __init__(self, cache_file):
        self.cache_file = cache_file
        from ultralytics.data.utils import load_dataset_cache_file
        self._cache = load_dataset_cache_file(cache_file)   # keep full dict
        self.labels = self._cache['labels']

    def save(self, save_path=None):
        """Save the (modified) cache back to disk.

        Args:
            save_path: Destination path. Defaults to the original cache file.
        """
        import numpy as np
        save_path = save_path or self.cache_file
        self._cache['labels'] = self.labels          # sync in-memory labels
        with open(str(save_path), "wb") as f:
            np.save(f, self._cache)
        print(f"[UltraCache] Saved {len(self.labels)} labels → {save_path}")



    @property
    def im_file_to_index(self):
        if not hasattr(self, '_im_file_to_index') or self._im_file_to_index is None:
            self._im_file_to_index={label['im_file']: idx for idx, label in enumerate(self.labels)}
        return self._im_file_to_index
    
    
    def __len__(self):
        return len(self.labels)

    def get_label_by_index(self, index):
        if index < 0 or index >= len(self.labels):
            raise IndexError(f"Index {index} is out of bounds for labels of length {len(self.labels)}")
        return self.labels[index]


    def get_label_by_im_file(self, im_file):
        if im_file not in self.im_file_to_index:
            raise KeyError(f"Image file '{im_file}' not found in cache.")
        index=self.im_file_to_index[im_file]
        return self.labels[index]

    def get_all_texts(self, unique=False) -> list:
        """Return all text strings across all labels.

        Each label's 'texts' is a list of single-item lists, e.g. [["dog"], ["cat"]].

        Args:
            unique: If True, return deduplicated list (order-preserving).

        Returns:
            list of str, e.g. ["umbrella", "bracelet", "dog", ...]
        """
        texts = []
        seen = set()
        for label in self.labels:
            for t in label.get("texts", []):
                name = t[0] if (t and len(t) > 0) else None
                if name is None:
                    continue
                if unique:
                    if name not in seen:
                        seen.add(name)
                        texts.append(name)
                else:
                    texts.append(name)
        return texts

    
    def to_ultra_result(self, index, save_path=None):
        """Convert a cached label at the given index to an ultralytics Results object.

        Args:
            index: Index of the label in the cache.
                save_path: Optional path to save the original image for visualization. If None, the image will not be saved.


        Returns:
            ultralytics.engine.results.Results object with orig_img, path, names, boxes.
        """
        import os
        import numpy as np
        import torch
        from PIL import Image
        from ultralytics.engine.results import Results
        from pathlib import Path
        from ultralytics.utils.ops import xywh2xyxy
        if isinstance(index, str) or isinstance(index, os.PathLike) or isinstance(index,Path):
            # If index is given as im_file path, convert to index
            im_file=str(index)
            if im_file not in self.im_file_to_index:
                raise KeyError(f"Image file '{im_file}' not found in cache.")
            index=self.im_file_to_index[im_file]

        label = self.get_label_by_index(index)

        # Resolve image path
        im_path = label['im_file']
        # Load image
        orig_img = np.array(Image.open(im_path).convert("RGB"))
        img_h, img_w = orig_img.shape[:2]

        # Build names mapping from texts (each text is a list like ["umbrella"])
        texts = label.get('texts', [])
        unique_names = []
        for t in texts:
            cls_name = str(t[0]) if (t and len(t) > 0) else "unknown"
            if cls_name not in unique_names:
                unique_names.append(cls_name)
        names = {i: n for i, n in enumerate(unique_names)}

        # Get bboxes and cls
        bboxes = np.array(label['bboxes'], dtype=np.float32)  # (N, 4) xywh
        cls = np.array(label['cls'], dtype=np.float32).reshape(-1)  # (N,)

        if bboxes.shape[0] == 0:
            boxes_tensor = torch.empty((0, 6), dtype=torch.float32)
        else:
            # Denormalize if needed
            if label.get('normalized', False):
                bboxes[:, [0, 2]] *= img_w
                bboxes[:, [1, 3]] *= img_h

            # Convert xywh to xyxy
            if label.get('bbox_format', 'xywh') == 'xywh':
                bboxes = xywh2xyxy(bboxes)

            # Build [x1, y1, x2, y2, conf, cls] per row
            n = bboxes.shape[0]
            confs = np.ones(n, dtype=np.float32)
            data = np.column_stack([bboxes, confs, cls])
            boxes_tensor = torch.from_numpy(data)

        result = Results(
            orig_img=orig_img,
            path=im_path,
            names=names,
            boxes=boxes_tensor,
        )

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result.save(save_path)

        return result
    


if __name__ == "__main__":
    cache_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache"
    
    texts= UltraCache(cache_file).get_all_texts(unique=True)

    import os
    os.makedirs("../buffer/temp", exist_ok=True)
    # save to ../buffer/temp/mixed_texts.txt
    with open("../buffer/temp/mixed_texts.txt", "w") as f:
        for t in texts:
            f.write(t + "\n")