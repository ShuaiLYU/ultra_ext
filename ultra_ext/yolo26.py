
import numpy as np


def yolo26_seg_flops():
    
    from ultralytics import YOLO


    model_weight="yolo26n-seg.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    model_weight="yolo26s-seg.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 
    
    model_weight="yolo26m-seg.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    model_weight="yolo26l-seg.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    model_weight="yolo26x-seg.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    # YOLO26n-seg summary (fused): 139 layers, 2,722,980 parameters, 0 gradients, 9.1 GFLOPs
    # YOLO26n-seg summary (fused): 175 layers, 3,115,632 parameters, 0 gradients, 9.1 GFLOPs
    # YOLO26s-seg summary (fused): 139 layers, 10,396,300 parameters, 0 gradients, 34.2 GFLOPs
    # YOLO26s-seg summary (fused): 175 layers, 11,485,496 parameters, 0 gradients, 34.2 GFLOPs
    # YOLO26m-seg summary (fused): 149 layers, 23,569,148 parameters, 0 gradients, 121.5 GFLOPs
    # YOLO26m-seg summary (fused): 185 layers, 27,081,384 parameters, 0 gradients, 121.5 GFLOPs
    # YOLO26l-seg summary (fused): 207 layers, 27,965,436 parameters, 0 gradients, 139.8 GFLOPs
    # YOLO26l-seg summary (fused): 243 layers, 31,477,672 parameters, 0 gradients, 139.8 GFLOPs
    # YOLO26x-seg summary (fused): 207 layers, 62,819,132 parameters, 0 gradients, 313.5 GFLOPs
    # YOLO26x-seg summary (fused): 243 layers, 70,637,032 parameters, 0 gradients, 313.5 GFLOPs

def yolo26_flops():
    
    from ultralytics import YOLO


    model_weight="yolo26n.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    model_weight="yolo26s.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 
    
    model_weight="yolo26m.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    model_weight="yolo26l.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    model_weight="yolo26x.pt"
    model=YOLO(model_weight)
    model.fuse()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.fuse() 

    # YOLO26n summary (fused): 122 layers, 2,408,932 parameters, 0 gradients, 5.4 GFLOPs
    # YOLO26n summary (fused): 146 layers, 2,562,496 parameters, 0 gradients, 5.4 GFLOPs
    # YOLO26s summary (fused): 122 layers, 9,496,140 parameters, 0 gradients, 20.7 GFLOPs
    # YOLO26s summary (fused): 146 layers, 9,990,792 parameters, 0 gradients, 20.7 GFLOPs
    # YOLO26m summary (fused): 132 layers, 20,411,132 parameters, 0 gradients, 68.2 GFLOPs
    # YOLO26m summary (fused): 156 layers, 21,868,152 parameters, 0 gradients, 68.2 GFLOPs
    # YOLO26l summary (fused): 190 layers, 24,807,420 parameters, 0 gradients, 86.4 GFLOPs
    # YOLO26l summary (fused): 214 layers, 26,264,440 parameters, 0 gradients, 86.4 GFLOPs
    # YOLO26x summary (fused): 190 layers, 55,725,948 parameters, 0 gradients, 193.9 GFLOPs
    # YOLO26x summary (fused): 214 layers, 58,940,472 parameters, 0 gradients, 193.9 GFLOPs