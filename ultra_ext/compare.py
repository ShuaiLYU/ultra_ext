
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

    # from ultralytics.utils.torch_utils import strip_optimizer
    # strip_optimizer(file2) # remove ema weight to avoid conflict


    # from pathlib import Path
    # from ultralytics.utils.torch_utils import strip_optimizer

    # strip_optimizer(file1)
    # strip_optimizer(file2)


    print("=== File Sizes ===")
    if os.path.exists(file1):
        size1 = os.path.getsize(file1) / (1024**2)  # MB
        print(f"Model 1 ({file1}): {size1:.2f} MB")
    if os.path.exists(file2):
        size2 = os.path.getsize(file2) / (1024**2)  # MB
        print(f"Model 2 ({file2}): {size2:.2f} MB")

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载两个模型文件
    model1 = torch.load(file1, weights_only=False, map_location=device)

    model2 = torch.load(file2, weights_only=False, map_location=device)

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



import numpy as np
from pathlib import Path


def _compare_values_recursive(val1, val2, path="root", max_depth=20, current_depth=0):
    """Recursively compare two values, handling nested structures.
    
    Args:
        val1: First value to compare
        val2: Second value to compare
        path: Current path in the data structure (for error reporting)
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        
    Returns:
        tuple: (is_equal: bool, difference_info: dict or None)
    """
    if current_depth > max_depth:
        return False, {
            'type': 'max_depth_exceeded',
            'path': path,
            'message': f'Maximum recursion depth {max_depth} exceeded'
        }
    
    # Check if types match
    if type(val1) != type(val2):
        return False, {
            'type': 'type_mismatch',
            'path': path,
            'type1': type(val1).__name__,
            'type2': type(val2).__name__
        }
    
    # Handle None
    if val1 is None and val2 is None:
        return True, None
    
    # Handle numpy arrays
    if isinstance(val1, np.ndarray):
        try:
            if val1.shape != val2.shape:
                return False, {
                    'type': 'numpy_array_shape_mismatch',
                    'path': path,
                    'shape1': val1.shape,
                    'shape2': val2.shape,
                    'dtype1': str(val1.dtype),
                    'dtype2': str(val2.dtype)
                }
            
            arrays_equal = np.array_equal(val1, val2)
            if not arrays_equal:
                return False, {
                    'type': 'numpy_array_content_mismatch',
                    'path': path,
                    'shape': val1.shape,
                    'dtype': str(val1.dtype),
                    'max_diff': float(np.max(np.abs(val1 - val2))) if np.issubdtype(val1.dtype, np.number) else 'N/A'
                }
            return True, None
        except Exception as e:
            return False, {
                'type': 'numpy_comparison_error',
                'path': path,
                'error': str(e)
            }
    
    # Handle torch tensors
    try:
        import torch
        if isinstance(val1, torch.Tensor):
            try:
                if val1.shape != val2.shape:
                    return False, {
                        'type': 'torch_tensor_shape_mismatch',
                        'path': path,
                        'shape1': tuple(val1.shape),
                        'shape2': tuple(val2.shape),
                        'dtype1': str(val1.dtype),
                        'dtype2': str(val2.dtype)
                    }
                
                tensors_equal = torch.equal(val1, val2)
                if not tensors_equal:
                    return False, {
                        'type': 'torch_tensor_content_mismatch',
                        'path': path,
                        'shape': tuple(val1.shape),
                        'dtype': str(val1.dtype),
                        'max_diff': float(torch.max(torch.abs(val1 - val2))) if val1.dtype in [torch.float32, torch.float64, torch.float16] else 'N/A'
                    }
                return True, None
            except Exception as e:
                return False, {
                    'type': 'torch_comparison_error',
                    'path': path,
                    'error': str(e)
                }
    except ImportError:
        pass  # torch not available
    
    # Handle dictionaries - recursively compare
    if isinstance(val1, dict):
        keys1 = set(val1.keys())
        keys2 = set(val2.keys())
        
        if keys1 != keys2:
            return False, {
                'type': 'dict_keys_mismatch',
                'path': path,
                'keys_only_in_1': keys1 - keys2,
                'keys_only_in_2': keys2 - keys1
            }
        
        # Recursively compare all values
        for key in keys1:
            is_equal, diff_info = _compare_values_recursive(
                val1[key], val2[key], 
                path=f"{path}.{key}",
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            if not is_equal:
                return False, diff_info
        
        return True, None
    
    # Handle lists - recursively compare elements
    if isinstance(val1, list):
        if len(val1) != len(val2):
            return False, {
                'type': 'list_length_mismatch',
                'path': path,
                'length1': len(val1),
                'length2': len(val2)
            }
        
        for i, (item1, item2) in enumerate(zip(val1, val2)):
            is_equal, diff_info = _compare_values_recursive(
                item1, item2,
                path=f"{path}[{i}]",
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            if not is_equal:
                return False, diff_info
        
        return True, None
    
    # Handle tuples - recursively compare elements
    if isinstance(val1, tuple):
        if len(val1) != len(val2):
            return False, {
                'type': 'tuple_length_mismatch',
                'path': path,
                'length1': len(val1),
                'length2': len(val2)
            }
        
        for i, (item1, item2) in enumerate(zip(val1, val2)):
            is_equal, diff_info = _compare_values_recursive(
                item1, item2,
                path=f"{path}[{i}]",
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            if not is_equal:
                return False, diff_info
        
        return True, None
    
    # Handle primitive types (int, float, str, bool, etc.)
    try:
        is_equal = val1 == val2
        
        # Handle the case where == returns an array (shouldn't happen here but be safe)
        if isinstance(is_equal, (np.ndarray, np.bool_)):
            is_equal = bool(is_equal.all() if hasattr(is_equal, 'all') else is_equal)
        
        if not is_equal:
            return False, {
                'type': f'{type(val1).__name__}_value_mismatch',
                'path': path,
                'value1': str(val1)[:200],  # Truncate long values
                'value2': str(val2)[:200]
            }
        return True, None
    except Exception as e:
        return False, {
            'type': 'comparison_error',
            'path': path,
            'value1_type': type(val1).__name__,
            'value2_type': type(val2).__name__,
            'error': str(e)
        }


def compare_cache_files(cache_path1: Path | str, cache_path2: Path | str, verbose: bool = True) -> dict:
    """Compare two cache files and return detailed comparison results.
    
    Args:
        cache_path1: Path to first cache file
        cache_path2: Path to second cache file
        verbose: If True, print comparison results
        
    Returns:
        dict: Comparison results with keys:
            - 'file_sizes_match': bool - Whether file sizes are the same
            - 'keys_match': bool - Whether cache keys are the same
            - 'lengths_match': bool - Whether number of items are the same
            - 'content_match': bool - Whether cache content is identical
            - 'size1': int - File size of first cache
            - 'size2': int - File size of second cache
            - 'keys1': set - Keys in first cache
            - 'keys2': set - Keys in second cache
            - 'keys_only_in_1': set - Keys only in first cache
            - 'keys_only_in_2': set - Keys only in second cache
            - 'length1': int - Number of items in first cache
            - 'length2': int - Number of items in second cache
            - 'differences': dict - Detailed differences for each key
    """
    cache_path1 = Path(cache_path1)
    cache_path2 = Path(cache_path2)
    
    result = {
        'file_sizes_match': False,
        'keys_match': False,
        'lengths_match': False,
        'content_match': False,
        'differences': {}
    }
    
    # Check if files exist
    if not cache_path1.exists():
        print(f"❌ Error: Cache file 1 not found: {cache_path1}")
        return result
    
    if not cache_path2.exists():
        print(f"❌ Error: Cache file 2 not found: {cache_path2}")
        return result
    
    try:
        # 1. Compare file sizes
        size1 = cache_path1.stat().st_size
        size2 = cache_path2.stat().st_size
        result['size1'] = size1
        result['size2'] = size2
        result['file_sizes_match'] = size1 == size2
        
        # 2. Load cache files
        cache1 = np.load(str(cache_path1), allow_pickle=True).item()
        cache2 = np.load(str(cache_path2), allow_pickle=True).item()
        
        # 3. Compare keys
        keys1 = set(cache1.keys())
        keys2 = set(cache2.keys())
        result['keys1'] = keys1
        result['keys2'] = keys2
        result['length1'] = len(keys1)
        result['length2'] = len(keys2)
        result['keys_match'] = keys1 == keys2
        result['lengths_match'] = len(keys1) == len(keys2)
        
        keys_only_in_1 = keys1 - keys2
        keys_only_in_2 = keys2 - keys1
        result['keys_only_in_1'] = keys_only_in_1
        result['keys_only_in_2'] = keys_only_in_2
        
        # 4. Compare content for common keys using recursive comparison
        content_match = True
        common_keys = keys1 & keys2
        
        for key in common_keys:
            val1 = cache1[key]
            val2 = cache2[key]
            
            # Use recursive comparison
            is_equal, diff_info = _compare_values_recursive(val1, val2, path=f"cache.{key}")
            
            if not is_equal:
                result['differences'][key] = diff_info
                content_match = False
        
        result['content_match'] = content_match and len(keys_only_in_1) == 0 and len(keys_only_in_2) == 0
        
        # 5. Print results if verbose
        if verbose:
            print("\n" + "=" * 80)
            print("CACHE FILE COMPARISON RESULTS")
            print("=" * 80)
            
            print(f"\nFile 1: {cache_path1}")
            print(f"File 2: {cache_path2}")
            
            # File sizes
            print(f"\n📊 File Sizes:")
            print(f"  Cache 1: {size1:,} bytes")
            print(f"  Cache 2: {size2:,} bytes")
            print(f"  Match: {'✅ YES' if result['file_sizes_match'] else '❌ NO'}")
            
            # Keys and length
            print(f"\n🔑 Keys & Length:")
            print(f"  Cache 1 keys: {result['length1']} ({keys1})")
            print(f"  Cache 2 keys: {result['length2']} ({keys2})")
            print(f"  Keys match: {'✅ YES' if result['keys_match'] else '❌ NO'}")
            print(f"  Length match: {'✅ YES' if result['lengths_match'] else '❌ NO'}")
            
            if keys_only_in_1:
                print(f"  ⚠️  Keys only in Cache 1: {keys_only_in_1}")
            if keys_only_in_2:
                print(f"  ⚠️  Keys only in Cache 2: {keys_only_in_2}")
            
            # Content comparison
            print(f"\n📦 Content:")
            if result['differences']:
                print(f"  Differences found in {len(result['differences'])} keys:")
                for key, diff in result['differences'].items():
                    print(f"    - {key}: {diff}")
            else:
                if result['content_match']:
                    print(f"  ✅ All common keys have identical content")
            
            # Overall result
            print(f"\n{'=' * 80}")
            if result['content_match'] and result['file_sizes_match']:
                print("✅ CACHES ARE IDENTICAL")
            elif result['keys_match'] and result['lengths_match']:
                print("⚠️  CACHES HAVE SAME STRUCTURE BUT DIFFERENT CONTENT")
            else:
                print("❌ CACHES ARE DIFFERENT (different structure)")
            print("=" * 80 + "\n")
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading cache files: {e}")
        return result


# compare the labels in the cache files to see if they are the same
def compare_cache_labels(cache_path1: Path | str, cache_path2: Path | str, verbose: bool = True) -> dict:
    """Compare labels in two cache files and return comparison results.
    
    Args:
        cache_path1: Path to first cache file
        cache_path2: Path to second cache file
        verbose: If True, print comparison results
    """
    cache_path1 = Path(cache_path1)
    cache_path2 = Path(cache_path2)
    
    result = {
        'labels_match': False,
        'labels1': None,
        'labels2': None,
        'differences': None
    }
    
    # Check if files exist
    if not cache_path1.exists():
        print(f"❌ Error: Cache file 1 not found: {cache_path1}")
        return result
    
    if not cache_path2.exists():
        print(f"❌ Error: Cache file 2 not found: {cache_path2}")
        return result
    
    try:
        # Load cache files
        cache1 = np.load(str(cache_path1), allow_pickle=True).item()
        cache2 = np.load(str(cache_path2), allow_pickle=True).item()
        
        # Extract labels (assuming they are under a key like 'labels' or similar)
        labels1 = cache1.get('labels', None)
        labels2 = cache2.get('labels', None)
        
        result['labels1'] = labels1
        result['labels2'] = labels2
        
        if labels1 is None or labels2 is None:
            print("⚠️  One or both caches do not contain 'labels' key")
            return result
        
        # Compare labels using recursive comparison
        is_equal, diff_info = _compare_values_recursive(labels1, labels2, path="cache.labels")
        
        result['labels_match'] = is_equal
        result['differences'] = diff_info
        
        # Print results if verbose
        if verbose:
            print("\n" + "=" * 80)
            print("CACHE LABELS COMPARISON RESULTS")
            print("=" * 80)
            
            print(f"\nFile 1: {cache_path1}")
            print(f"File 2: {cache_path2}")
            
            if is_equal:
                print("\n✅ Labels match exactly")
            else:
                print("\n❌ Labels do NOT match:")
                print(f"Difference info: {diff_info}")
            
            print("=" * 80 + "\n")
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading cache files: {e}")
        return result




"""
比较两个模型的网络结构，输出 CSV 表格，并打印不一样的层。
"""
import csv
import torch
from collections import Counter


def compare_arch(pt1: str, pt2: str, out_csv: str = None):
    def get_arch(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = (ckpt.get("model") or ckpt.get("ema")) if isinstance(ckpt, dict) else ckpt
        assert m is not None, f"{path}: 找不到 model"
        sd = m.state_dict()
        return {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items()}

    print(f"Loading pt1: {pt1}")
    arch1 = get_arch(pt1)
    print(f"Loading pt2: {pt2}")
    arch2 = get_arch(pt2)

    all_keys = sorted(arch1.keys() | arch2.keys())

    rows = []
    for k in all_keys:
        in1, in2 = k in arch1, k in arch2
        shape1, dtype1 = arch1[k] if in1 else ("—", "—")
        shape2, dtype2 = arch2[k] if in2 else ("—", "—")

        if   not in1:          status = "only_in_2"
        elif not in2:          status = "only_in_1"
        elif shape1 != shape2: status = "shape_diff"
        elif dtype1 != dtype2: status = "dtype_diff"
        else:                  status = "same"

        rows.append(dict(key=k,
                         shape_1=str(shape1), dtype_1=dtype1,
                         shape_2=str(shape2), dtype_2=dtype2,
                         status=status))
    if out_csv is not None:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["key","shape_1","dtype_1","shape_2","dtype_2","status"])
            w.writeheader()
            w.writerows(rows)

    cnt = Counter(r["status"] for r in rows)
    if out_csv is not None:
        print(f"\n✅ 已写出: {out_csv}  ({len(rows)} 行)")
    print(f"   same       : {cnt['same']}")
    print(f"   shape_diff : {cnt['shape_diff']}")
    print(f"   dtype_diff : {cnt['dtype_diff']}")
    print(f"   only_in_1  : {cnt['only_in_1']}  (pt1 独有，pt2 缺失)")
    print(f"   only_in_2  : {cnt['only_in_2']}  (pt2 独有，pt1 缺失)")

    # 打印不一样的层
    diff_rows = [r for r in rows if r["status"] != "same"]
    if diff_rows:
        print(f"\n{'─'*130}")
        print(f"{'差异层列表':^130}")
        print(f"{'─'*130}")
        print(f"{'status':<12} {'key':<70} {'shape_1':<22} {'shape_2':<22} {'dtype_1':<14} {'dtype_2'}")
        print(f"{'─'*130}")
        # 按 status 分组排序：only_in_1 / only_in_2 / shape_diff / dtype_diff
        order = {"only_in_1": 0, "only_in_2": 1, "shape_diff": 2, "dtype_diff": 3}
        for r in sorted(diff_rows, key=lambda x: (order.get(x["status"], 9), x["key"])):
            print(f"{r['status']:<12} {r['key']:<70} {r['shape_1']:<22} {r['shape_2']:<22} {r['dtype_1']:<14} {r['dtype_2']}")
        print(f"{'─'*130}")
    else:
        print("\n✅ 两个模型结构完全一致，无差异层。")

    # 打印 train_args 里记录的原始 yaml
    ckpt1 = torch.load(pt1, map_location="cpu", weights_only=False)
    if isinstance(ckpt1, dict) and "train_args" in ckpt1:
        print(f"\n📋 pt1 训练时使用的 yaml: {ckpt1['train_args'].get('model', '未记录')}")




def count_bias_modlules(pt):
    """
    Count how many modules have bias parameters, split by conv vs BN.

    Args:
        model: nn.Module (or a loaded checkpoint dict with key 'model')

    Returns:
        dict with keys: total, conv, bn, other
    """
    import torch.nn as nn

    model = torch.load(pt, weights_only=False, map_location="cpu")

    if isinstance(model, dict):
        model = model.get("model") or model.get("ema")

    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                  nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    bn_types   = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                  nn.InstanceNorm2d, nn.InstanceNorm3d)

    total = conv_count = bn_count = other_count = 0

    for name, module in model.named_modules():
        bias = getattr(module, "bias", None)
        if bias is None:
            continue
        # bias exists but may be disabled (set to None via bias=False)
        if not isinstance(bias, nn.Parameter):
            continue

        total += 1
        if isinstance(module, conv_types):
            conv_count += 1
        elif isinstance(module, bn_types):
            bn_count += 1
        else:
            other_count += 1

    result = dict(total=total, conv=conv_count, bn=bn_count, other=other_count)

    print("=== Bias Module Count ===")
    print(f"  Total modules with bias : {total}")
    print(f"  Conv  (Conv1/2/3d etc.) : {conv_count}")
    print(f"  BN    (BN/GN/LN etc.)  : {bn_count}")
    print(f"  Other                   : {other_count}")

    return result
if __name__ == "__main__":


    pass