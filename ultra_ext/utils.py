"""
Utility functions for YOLO result processing.
"""

import os
import torch
from copy import deepcopy
from typing import Dict


def split_detections_by_class(result) -> Dict[int, object]:
    """
    Split a YOLO detection result into separate results grouped by class.
    
    Args:
        result: YOLO detection result object
        
    Returns:
        dict: Dictionary mapping class_id -> result containing only that class
              Example: {0: result_with_person, 1: result_with_bicycle, ...}
    """
    if len(result.boxes) == 0:
        return {}
    
    # Get all unique class IDs
    classes = result.boxes.cls.cpu().numpy()
    unique_cls = sorted(set(int(c) for c in classes))
    
    split_results = {}
    
    for cls_id in unique_cls:
        # Find indices for this class
        indices = [i for i, c in enumerate(classes) if int(c) == cls_id]
        
        if len(indices) == 0:
            continue
        
        # Create a new result for this class
        new_result = deepcopy(result)
        
        # Extract boxes for this class (N, 4)
        boxes_xyxy = result.boxes.xyxy[indices].cpu()
        scores = result.boxes.conf[indices].cpu().unsqueeze(1)
        cls_tensor = result.boxes.cls[indices].cpu().unsqueeze(1)
        
        # Concatenate to create full boxes tensor [x1, y1, x2, y2, conf, cls]
        boxes_with_conf_cls = torch.cat([boxes_xyxy, scores, cls_tensor], dim=1)
        
        # Extract masks for this class if available
        masks_tensor = result.masks.data[indices] if result.masks is not None else None
        
        # Update the new result
        new_result.update(boxes=boxes_with_conf_cls, masks=masks_tensor)
        
        split_results[cls_id] = new_result
    
    return split_results


def keep_best_detection_per_class(result) -> object:
    """
    Keep only the highest confidence detection for each class (class-wise NMS).
    
    For each unique class in the result, retains only the detection with the
    highest confidence score. Useful for single-instance-per-class scenarios.
    
    Args:
        result: YOLO detection result object
    
    Returns:
        Filtered result with maximum one detection per class
    """
    if len(result.boxes) == 0:
        return result
    
    # Get current class IDs and scores
    classes = result.boxes.cls.cpu().numpy()  # (N,)
    scores = result.boxes.conf.cpu().numpy()  # (N,)
    
    # Find best detection index for each class
    best_indices = {}
    for i, (cls_id, score) in enumerate(zip(classes, scores)):
        cls_id_int = int(cls_id)
        if cls_id_int not in best_indices or score > scores[best_indices[cls_id_int]]:
            best_indices[cls_id_int] = i
    
    # Extract indices of best detections
    selected_indices = sorted(best_indices.values())
    
    # Create filtered result with only the best detections
    filtered_result = deepcopy(result)
    filtered_result.boxes = result.boxes[selected_indices]
    if result.masks is not None:
        filtered_result.masks = result.masks[selected_indices]
    
    return filtered_result


def open_in_vscode(file_path: str):
    """
    Open the specified file in VSCode.
    
    Args:
        file_path: Path to the file to open
    """
    import subprocess
    try:
        subprocess.run(["code", file_path], check=True)
    except Exception as e:
        print(f"Failed to open {file_path} in VSCode: {e}")


def save_res(res, save_path: str="./runs/temp/res.jpg",vscode_open: bool=False):
    """
    Save the image to the specified path and open it in VSCode.
    Args:
        res: Result object to save
        save_path: Path to save the image
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    res.save(save_path)
    print(f"Saved visualization to {save_path}")
    if vscode_open:
        open_in_vscode(save_path)
          
def save_results_grid(results, save_path: str="./runs/temp/results_grid.jpg", cols: int = None,vscode_open: bool=False):
    """
    Save multiple result images as a grid layout to a single image file.
    
    Args:
        results: List of YOLO result objects
        save_path: Path to save the concatenated image
        cols: Number of columns in grid. If None, auto-calculates based on sqrt(n)
    
    Returns:
        str: Path where the image was saved
    """
    import cv2
    import numpy as np
    from pathlib import Path
    import math
    
    if not results:
        print("No results to save")
        return None
    
    # Plot all results
    ims = [res.plot(show=False) for res in results]
    n = len(ims)
    
    # Auto-calculate grid layout
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    # Get max dimensions
    heights = [im.shape[0] for im in ims]
    widths = [im.shape[1] for im in ims]
    max_h, max_w = max(heights), max(widths)
    
    # Resize all images to same size for uniform grid
    resized_ims = []
    for im in ims:
        if im.shape[0] != max_h or im.shape[1] != max_w:
            im = cv2.resize(im, (max_w, max_h))
        resized_ims.append(im)
    
    # Pad with blank images if needed
    blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    while len(resized_ims) < rows * cols:
        resized_ims.append(blank)
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_ims = resized_ims[i * cols:(i + 1) * cols]
        grid_rows.append(np.hstack(row_ims))
    grid = np.vstack(grid_rows)
    
    # Save
    save_path = Path(save_path).absolute()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), grid)
    print(f"Saved {n} results grid ({rows}x{cols}) to {save_path}")
    if vscode_open:
        open_in_vscode(str(save_path))
    return str(save_path)
