"""
Utility functions for YOLO result processing.
"""

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