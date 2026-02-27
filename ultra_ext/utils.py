"""
Utility functions for YOLO result processing.
"""

import os
import torch
from copy import deepcopy
from typing import Dict
import cv2
import numpy as np
from pathlib import Path
import math

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
    if isinstance(res, np.ndarray):
        res=Image.fromarray(res)

    res.save(save_path)
    print(f"Saved visualization to {save_path}")
    if vscode_open:
        open_in_vscode(save_path)
          
def save_results_grid(results, save_path: str="./runs/temp/results_grid.jpg", cols: int = None,vscode_open: bool=False):
    """d
    Save multiple result images as a grid layout to a single image file.
    
    Args:
        results: List of YOLO result objects
        save_path: Path to save the concatenated image
        cols: Number of columns in grid. If None, auto-calculates based on sqrt(n)
    
    Returns:
        str: Path where the image was saved
    """

    
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


# a function read all shapes from labelme json file

def read_labelme_shapes(labelme_json_path: str):
    """
    Read shapes from a LabelMe JSON file. And set 
    
    Args:
        labelme_json_path: Path to the LabelMe JSON file. 
    Returns:
        List of shape dictionaries from the JSON file
    """

    import json
    import numpy as np
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    shapes=data.get("shapes", [])

    def points2bbox(points):
        points = np.array(points)
        min_x = int(np.min(points[:, 0]))
        max_x = int(np.max(points[:, 0]))
        min_y = int(np.min(points[:, 1]))
        max_y = int(np.max(points[:, 1]))
        return [min_x, min_y, max_x, max_y]
    
    for shape in shapes:
        shape["bbox"] = points2bbox(shape["points"])
    return shapes

import cv2
def draw_bboxes_labels_on_img(img, bboxes, labels):
    """
    Display bounding boxes and labels on the image.
    
    Args:
        image_path (str): Path to the image file.
        bboxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        labels (list): List of labels corresponding to the bounding boxes.
    """
    img=img.copy()
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


from PIL import Image
import numpy as np
def read_im_rgb(img_path):
    return np.array(Image.open(img_path).convert("RGB"))



import torch
import os


def compare_weights(pt1, pt2):
    """
    Compare two PyTorch weight files and analyze their differences.
    
    Args:
        pt1: Path to the first PyTorch weight file
        pt2: Path to the second PyTorch weight file
    """
    file1 = pt1
    file2 = pt2
    
    # strip_optimizer(file1)
    # strip_optimizer(file2)


    print("=== File Sizes ===")

    from ultralytics.utils.torch_utils import strip_optimizer
    strip_optimizer(file2) # remove ema weight to avoid conflict


    from pathlib import Path
    from ultralytics.utils.torch_utils import strip_optimizer

    # strip_optimizer(file1)
    # strip_optimizer(file2)


    print("=== File Sizes ===")
    if os.path.exists(file1):
        size1 = os.path.getsize(file1) / (1024**2)  # MB
        print(f"Model 1 ({file1}): {size1:.2f} MB")
    if os.path.exists(file2):
        size2 = os.path.getsize(file2) / (1024**2)  # MB
        print(f"Model 2 ({file2}): {size2:.2f} MB")

    # 加载两个模型文件
    model1 = torch.load(file1, weights_only=False)
    model2 = torch.load(file2, weights_only=False)

    # 1. Check keys contained in files (determine if training state exists)
    print("=== Model 1 Contents ===")
    for k in model1.keys():
        print(k)

    print("\n=== Model 2 Contents ===")
    for k in model2.keys():
        print(k)


    assert "model" in model1, f"Model 1 ({file1}) missing 'model' key, please check file contents"
    assert model1["model"] is not None, f"Model 1 ({file1}) 'model' is None, please check file contents"
    assert "model" in model2, f"Model 2 ({file2}) missing 'model' key, please check file contents"
    assert model2["model"] is not None, f"Model 2 ({file2}) 'model' is None, please check file contents"

    # 2. Check model parameter count (determine if structure differs)
    if "model" in model1 and "model" in model2 and model1["model"] is not None and model2["model"] is not None:
        params1 = sum(p.numel() for p in model1["model"].parameters())
        params2 = sum(p.numel() for p in model2["model"].parameters())
        print(f"\nModel 1 parameter count: {params1 / 1e6:.2f} M")
        print(f"Model 2 parameter count: {params2 / 1e6:.2f} M")
    else:
        print("\n⚠️ Cannot calculate parameter count:")
        if "model" not in model1:
            print(f"  - Model 1 ({file1}) missing 'model' key")
        elif model1["model"] is None:
            print(f"  - Model 1 ({file1}) 'model' is None")
        if "model" not in model2:
            print(f"  - Model 2 ({file2}) missing 'model' key")
        elif model2["model"] is None:
            print(f"  - Model 2 ({file2}) 'model' is None")

    # 3. Check model parameter data types (float32 or float16)
    print("\n=== Model 1 Parameter Data Types ===")
    if "model" in model1 and model1["model"] is not None:
        dtypes1 = set(p.dtype for p in model1["model"].parameters())
        for dtype in dtypes1:
            print(f"Data type: {dtype}")
    else:
        if "model" not in model1:
            print(f"⚠️ Model 1 ({file1}) missing 'model' key")
        elif model1["model"] is None:
            print(f"⚠️ Model 1 ({file1}) 'model' is None")

    print("\n=== Model 2 Parameter Data Types ===")
    if "model" in model2 and model2["model"] is not None:
        dtypes2 = set(p.dtype for p in model2["model"].parameters())
        for dtype in dtypes2:
            print(f"Data type: {dtype}")
    else:
        if "model" not in model2:
            print(f"⚠️ Model 2 ({file2}) missing 'model' key")
        elif model2["model"] is None:
            print(f"⚠️ Model 2 ({file2}) 'model' is None")

    # 4. Check component sizes (find reason for file size difference)
    print("\n=== Model 1 Component Sizes ===")
    for key in model1.keys():
        if key == "model" or key == "ema":
            if model1[key] is not None:
                params = sum(p.numel() * p.element_size() for p in model1[key].parameters())
                print(f"{key}: {params / (1024**2):.2f} MB")
        elif key == "optimizer":
            if model1[key] is not None:
                # 计算optimizer state的大小
                opt_size = 0
                if "state" in model1[key]:
                    for state in model1[key]["state"].values():
                        for v in state.values():
                            if torch.is_tensor(v):
                                opt_size += v.numel() * v.element_size()
                print(f"{key}: {opt_size / (1024**2):.2f} MB")
        else:
            # 其他小对象
            pass

    print("\n=== Model 2 Component Sizes ===")
    for key in model2.keys():
        if key == "model" or key == "ema":
            if model2[key] is not None:
                params = sum(p.numel() * p.element_size() for p in model2[key].parameters())
                print(f"{key}: {params / (1024**2):.2f} MB")
        elif key == "optimizer":
            if model2[key] is not None:
                # 计算optimizer state的大小
                opt_size = 0
                if "state" in model2[key]:
                    for state in model2[key]["state"].values():
                        for v in state.values():
                            if torch.is_tensor(v):
                                opt_size += v.numel() * v.element_size()
                print(f"{key}: {opt_size / (1024**2):.2f} MB")
        else:
            # 其他小对象
            pass

    # 5. Detailed comparison of model parameters and buffers (find reason for size difference)
    print("\n=== Detailed Model Size Difference Analysis ===")

    # Check parameters
    if "model" in model1 and "model" in model2 and model1["model"] is not None and model2["model"] is not None:
        # Parameter sizes
        params_size1 = sum(p.numel() * p.element_size() for p in model1["model"].parameters())
        params_size2 = sum(p.numel() * p.element_size() for p in model2["model"].parameters())
        
        # Buffer sizes (buffers are not parameters but also take space)
        buffers_size1 = sum(b.numel() * b.element_size() for b in model1["model"].buffers())
        buffers_size2 = sum(b.numel() * b.element_size() for b in model2["model"].buffers())
        
        print(f"Model 1 - Parameters: {params_size1 / (1024**2):.2f} MB, Buffers: {buffers_size1 / (1024**2):.2f} MB")
        print(f"Model 2 - Parameters: {params_size2 / (1024**2):.2f} MB, Buffers: {buffers_size2 / (1024**2):.2f} MB")
        
        # Check total size of all tensors in state_dict
        state_dict1 = model1["model"].state_dict()
        state_dict2 = model2["model"].state_dict()
        
        total_size1 = sum(v.numel() * v.element_size() for v in state_dict1.values() if torch.is_tensor(v))
        total_size2 = sum(v.numel() * v.element_size() for v in state_dict2.values() if torch.is_tensor(v))
        
        print(f"\nstate_dict total size - Model 1: {total_size1 / (1024**2):.2f} MB, Model 2: {total_size2 / (1024**2):.2f} MB")
        
        # Check if there are different keys
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())
        
        if keys1 != keys2:
            print("\nDifferent keys found:")
            only_in_1 = keys1 - keys2
            only_in_2 = keys2 - keys1
            if only_in_1:
                print(f"Only in Model 1: {only_in_1}")
            if only_in_2:
                print(f"Only in Model 2: {only_in_2}")
        
        # Compare tensor sizes for common keys
        print("\nChecking shape differences for common keys:")
        diff_found = False
        for key in keys1 & keys2:
            if torch.is_tensor(state_dict1[key]) and torch.is_tensor(state_dict2[key]):
                if state_dict1[key].shape != state_dict2[key].shape:
                    print(f"  {key}: Model 1={state_dict1[key].shape}, Model 2={state_dict2[key].shape}")
                    diff_found = True
                elif state_dict1[key].dtype != state_dict2[key].dtype:
                    print(f"  {key}: Model 1 dtype={state_dict1[key].dtype}, Model 2 dtype={state_dict2[key].dtype}")
                    diff_found = True
        
        if not diff_found:
            print("  All common keys have consistent shapes and data types")
        
        # 6. Analyze which layers were trained/updated (weights changed)
        print("\n=== Analyzing Trained Layers (Weight Difference Analysis) ===")
        
        common_keys = keys1 & keys2
        trained_layers = []
        unchanged_layers = []
        
        for key in sorted(common_keys):
            if torch.is_tensor(state_dict1[key]) and torch.is_tensor(state_dict2[key]):
                # Ensure shapes match
                if state_dict1[key].shape != state_dict2[key].shape:
                    continue
                
                # Skip non-floating-point tensors (like indices, integers)
                if not state_dict1[key].dtype.is_floating_point:
                    continue
                
                # Calculate weight differences
                diff = (state_dict1[key].float() - state_dict2[key].float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                # Calculate relative difference (normalized)
                weight_scale = state_dict1[key].abs().mean().item() + 1e-8
                relative_diff = mean_diff / weight_scale
                
                # Determine if layer was trained (threshold is adjustable)
                if max_diff > 1e-6 or mean_diff > 1e-7:
                    trained_layers.append({
                        'name': key,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'relative_diff': relative_diff,
                        'shape': state_dict1[key].shape
                    })
                else:
                    unchanged_layers.append(key)
        
        # Sort by difference from largest to smallest
        trained_layers.sort(key=lambda x: x['mean_diff'], reverse=True)
        
        print(f"\nNumber of trained layers: {len(trained_layers)} / {len(common_keys)}")
        print(f"Number of unchanged layers: {len(unchanged_layers)}")
        
        if trained_layers:
            print("\nTrained layers (sorted by average difference):")
            print(f"{'Layer Name':<80} {'Shape':<25} {'Max Diff':<15} {'Mean Diff':<15} {'Relative Diff':<10}")
            print("=" * 150)
            
            for i, layer in enumerate(trained_layers[:50]):  # Show only first 50
                print(f"{layer['name']:<80} {str(layer['shape']):<25} "
                    f"{layer['max_diff']:<15.6e} {layer['mean_diff']:<15.6e} {layer['relative_diff']:<10.6f}")
            
            if len(trained_layers) > 50:
                print(f"\n... and {len(trained_layers) - 50} more trained layers")
            
            # Group statistics by layer type
            print("\nStatistics by layer type:")
            layer_groups = {}
            for layer in trained_layers:
                # Extract layer type (e.g. model.22.cv4.0.conv.weight -> cv4)
                parts = layer['name'].split('.')
                if len(parts) >= 3:
                    layer_type = parts[2]  # Usually cv2, cv3, cv4, savpe, etc.
                else:
                    layer_type = 'other'
                
                if layer_type not in layer_groups:
                    layer_groups[layer_type] = []
                layer_groups[layer_type].append(layer)
            
            for layer_type, layers in sorted(layer_groups.items()):
                avg_diff = sum(l['mean_diff'] for l in layers) / len(layers)
                print(f"  {layer_type}: {len(layers)} layers, average difference: {avg_diff:.6e}")
        
        if unchanged_layers and len(unchanged_layers) <= 20:
            print(f"\nUnchanged layers:")
            for key in unchanged_layers:
                print(f"  {key}")
        elif unchanged_layers:
            print(f"\nSample of unchanged layers (total {len(unchanged_layers)} layers):")
            for key in unchanged_layers[:10]:
                print(f"  {key}")
            print(f"  ... and {len(unchanged_layers) - 10} more unchanged layers")

        # 7. Specifically check cv4 bias and logit_scale parameters
        print("\n=== CV4 Layer bias and logit_scale Comparison ===")
        
        # Dynamically find detection head indices (usually the last layer)
        head_indices = set()
        for key in state_dict1.keys():
            if "cv4" in key or "one2one_cv4" in key:
                parts = key.split('.')
                if len(parts) >= 2 and parts[0] == "model":
                    head_indices.add(int(parts[1]))
        
        if not head_indices:
            print("No cv4 layers found")
        else:
            print(f"Detected head indices: {sorted(head_indices)}")
        
        cv4_params = {}
        for model_name, state_dict in [("Model 1 (before)", state_dict1), ("Model 2 (after)", state_dict2)]:
            print(f"\n{model_name}:")
            
            # 遍历所有检测头索引
            for head_idx in sorted(head_indices):
                # 检查 cv4 和 one2one_cv4
                for cv4_name in ["cv4", "one2one_cv4"]:
                    for i in range(3):  # 0, 1, 2
                        bias_key = f"model.{head_idx}.{cv4_name}.{i}.bias"
                        logit_scale_key = f"model.{head_idx}.{cv4_name}.{i}.logit_scale"
                        
                        # 打印 bias
                        if bias_key in state_dict:
                            bias_value = state_dict[bias_key]
                            print(f"  {cv4_name}[{i}].bias: {bias_value.item():.6f}")
                            if model_name not in cv4_params:
                                cv4_params[model_name] = {}
                            cv4_params[model_name][bias_key] = bias_value.item()
                        
                        # 打印 logit_scale
                        if logit_scale_key in state_dict:
                            logit_scale_value = state_dict[logit_scale_key]
                            print(f"  {cv4_name}[{i}].logit_scale: {logit_scale_value.item():.6f}")
                            if model_name not in cv4_params:
                                cv4_params[model_name] = {}
                            cv4_params[model_name][logit_scale_key] = logit_scale_value.item()
        
        # Calculate changes
        if len(cv4_params) == 2:
            print("\nChange Analysis:")
            model1_name = "Model 1 (before)"
            model2_name = "Model 2 (after)"
            
            all_keys = set(cv4_params[model1_name].keys()) | set(cv4_params[model2_name].keys())
            
            for key in sorted(all_keys):
                if key in cv4_params[model1_name] and key in cv4_params[model2_name]:
                    val1 = cv4_params[model1_name][key]
                    val2 = cv4_params[model2_name][key]
                    diff = val2 - val1
                    rel_change = (diff / abs(val1)) * 100 if val1 != 0 else float('inf')
                    
                    param_name = key.split('.')[-1]  # bias or logit_scale
                    layer_name = '.'.join(key.split('.')[-3:-1])  # cv4.0, cv4.1, etc.
                    
                    if abs(diff) > 1e-6:
                        print(f"  {layer_name}.{param_name}:")
                        print(f"    Before: {val1:.6f} → After: {val2:.6f}")
                        # print(f"    Change: {diff:+.6f} ({rel_change:+.2f}%)")
                    else:
                        print(f"  {layer_name}.{param_name}: unchanged ({val1:.6f})")

    else:
        print("Cannot compare model parameters, possibly missing 'model' key or model is None")    
        
    print("\n=== Analysis Complete ===")