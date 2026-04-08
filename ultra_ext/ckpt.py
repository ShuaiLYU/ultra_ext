import os



def sprint_ckpt(ckpt_path,wo_model=True,wo_ema=True):

    import torch

    ckpt= torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if wo_model:
        del ckpt["model"]
        print(f"ckpt keys (model removed): {list(ckpt.keys())}")
    if wo_ema:
        del ckpt["ema"]
        print(f"ckpt keys (EMA removed): {list(ckpt.keys())}")
    from ultra_ext.utils import super_print
    super_print("ckpt",ckpt)









def compare_model_and_ema(ckpt_path):

    import torch

    ckpt= torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model_obj = ckpt.get("model")
    ema_obj = ckpt.get("ema")

    if model_obj is None or ema_obj is None:
        if model_obj is None:
            print("Model not found in checkpoint.")
        if ema_obj is None:
            print("EMA not found in checkpoint.")
        return

    # --- Module-level comparison ---
    model_modules = {name: type(module).__name__ for name, module in model_obj.named_modules()}
    ema_modules   = {name: type(module).__name__ for name, module in ema_obj.named_modules()}

    only_in_model = set(model_modules) - set(ema_modules)
    only_in_ema   = set(ema_modules) - set(model_modules)
    type_mismatch = {n for n in model_modules.keys() & ema_modules.keys()
                     if model_modules[n] != ema_modules[n]}

    print("=== Module Structure ===")
    if only_in_model:
        print(f"Modules only in Model: {only_in_model}")
    if only_in_ema:
        print(f"Modules only in EMA:   {only_in_ema}")
    if type_mismatch:
        for n in sorted(type_mismatch):
            print(f"Type mismatch @ {n}: model={model_modules[n]}, ema={ema_modules[n]}")
    if not (only_in_model or only_in_ema or type_mismatch):
        print(f"Module structures are identical ({len(model_modules)} modules).")

    # --- State-dict key comparison ---
    model_state_dict = model_obj.state_dict()
    ema_state_dict   = ema_obj.state_dict()

    model_keys = set(model_state_dict.keys())
    ema_keys   = set(ema_state_dict.keys())

    missing_in_model = ema_keys - model_keys
    missing_in_ema   = model_keys - ema_keys

    print("\n=== State Dict Keys ===")
    if missing_in_model:
        print(f"Keys in EMA but missing in Model: {missing_in_model}")
    else:
        print("No keys are missing in Model compared to EMA.")

    if missing_in_ema:
        print(f"Keys in Model but missing in EMA: {missing_in_ema}")
    else:
        print("No keys are missing in EMA compared to Model.")

    # --- Weight value comparison ---
    print("\n=== Weight Value Differences (Model vs EMA) ===")
    import torch
    common_keys = model_keys & ema_keys
    diff_layers = []
    identical = 0

    for key in sorted(common_keys):
        t1 = model_state_dict[key]
        t2 = ema_state_dict[key]
        if not t1.dtype.is_floating_point:
            continue
        if t1.shape != t2.shape:
            continue
        diff = (t1.float() - t2.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        if max_diff > 1e-6:
            diff_layers.append((key, max_diff, mean_diff, tuple(t1.shape)))
        else:
            identical += 1

    if diff_layers:
        diff_layers.sort(key=lambda x: x[1], reverse=True)
        print(f"  {len(diff_layers)} layers differ, {identical} are identical.")
        print(f"\n  {'Key':<70} {'Shape':<20} {'Max Diff':<15} {'Mean Diff'}")
        print(f"  {'─'*120}")
        for key, max_d, mean_d, shape in diff_layers[:50]:
            print(f"  {key:<70} {str(shape):<20} {max_d:<15.6e} {mean_d:.6e}")
        if len(diff_layers) > 50:
            print(f"  ... and {len(diff_layers) - 50} more differing layers")
    else:
        print(f"  All {identical} floating-point layers are numerically identical between Model and EMA.")


if __name__ == "__main__":

    scale="26n"
    model_dir="./weights/yoloe26_weight/yoloe26_vp_seg"
    model_weight=f"{model_dir}/yoloe-{scale}-seg.pt"

    sprint_ckpt(model_weight,wo_model=True)
    # compare_model_and_ema(model_weight)