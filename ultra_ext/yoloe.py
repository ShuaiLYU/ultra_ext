



import torch
from ultralytics import YOLOE


def print_model_head_cv3(model_weight):

    model=YOLOE(model_weight)
    model.args['clip_weight_name']="mobileclip2:b"
    model_head=model.model.model[-1]

    for m in model_head.cv3:
        print(m)


def print_model_head_cv4(model_weight):
    print("*"*80)
    print(f"Printing model head cv4 for {model_weight}")
    model=YOLOE(model_weight)
    model.args['clip_weight_name']="mobileclip2:b"
    model_head=model.model.model[-1]
    for m in model_head.cv4:

        #print(m.norm)# batch norm layer
        print("BatchNorm running_mean:")
        print(torch.max(m.norm.running_mean), torch.min(m.norm.running_mean), torch.mean(m.norm.running_mean))
        print("BatchNorm running_var:")
        print(torch.max(m.norm.running_var), torch.min(m.norm.running_var), torch.mean(m.norm.running_var))
        print("BatchNorm weight:")
        print(torch.max(m.norm.weight), torch.min(m.norm.weight), torch.mean(m.norm.weight))
        print("BatchNorm bias:")
        print(torch.max(m.norm.bias), torch.min(m.norm.bias), torch.mean(m.norm.bias))


        print(f"logit_scale: {m.logit_scale.item()}")
        print(f"bias: {m.bias}")