
from pyexpat import model

import numpy as np





yoloe_train_data = {
	"val": {
		"yolo_data": [
			"../datasets/lvis.yaml"
		]
	},
	"train": {
		"yolo_data": [
			"../datasets/Objects365v1.yaml"
		],
		"grounding_data": [
			{
				"img_path": "../datasets/flickr/full_images/",
				"json_file": "../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json"
			},
			{
				"img_path": "../datasets/mixed_grounding/gqa/images",
				"json_file": "../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json"
			}
		]
	}
}



yoloe26_train_data = {
	"val": {
		"yolo_data": [
			"/home/louis/ultra_louis_work/datasets/lvis.yaml"
		]
	},
	
	"train": {
		"grounding_data": [
			{
				"img_path": "../datasets/flickr/full_images/",
				"json_file": "../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json"
			},
			{
				"img_path": "../datasets/mixed_grounding/gqa/images",
				"json_file": "../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json"
			},
			{
				"img_path": "../datasets/Objects365v1/images/train",
				"json_file": "../datasets/Objects365v1/annotations/objects365_train_segm.json"
			}
		]
	}
}



class TestSample:
	visual_prompts = [
		  
		{
			"image": "ultralytics/assets/bus.jpg",
			"prompts": dict(
				bboxes=np.array([
					[221.52, 405.8, 344.98, 857.54],  # Box enclosing person
					[120, 425, 160, 445],			  # Box enclosing glasses
				]),
				cls=np.array([
					0,  # ID to be assigned for person
					1,  # ID to be assigned for glasses
				]),
			),
		},
		{
			"image": "../datasets/coco/images/val2017/000000002157.jpg",
			"prompts": {
				"bboxes": np.array([[214, 184, 235, 203]]),
				"cls": np.array([0]),
			},
		},
		{
			"image": "../datasets/coco/images/val2017/000000002299.jpg",
			"prompts": {
				"bboxes": np.array([[162, 29, 208, 89]]),
				"cls": np.array([0]),
			},
		},
		{
			"image": "../datasets/coco/images/val2017/000000002149.jpg",
			"prompts": {
				"bboxes": np.array([[97, 98, 340, 321]]),
				"cls": np.array([0]),
			},
		},
		{
			"image": "../datasets/coco/images/val2017/000000005001.jpg",
			"prompts": {
				# "bboxes": np.array([[74, 232, 196, 470]]),
				"bboxes": np.array([[376, 93, 521, 468]]),
				"cls": np.array([0]),
			},
		},
		{
			"image": "../ultralytics/ultralytics/assets/bus.jpg",
			"prompts": {
				"bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]),
				"cls": np.array([0]),
			},
		},
		{
			"image": "../ultralytics/ultralytics/assets/zidane.jpg",
			"prompts": {
				"bboxes": np.array([[734, 38, 1144, 717]]),
				"cls": np.array([0]),
			},
		},
		{
			"image": "../datasets/coco/images/val2017/000000005992.jpg",
			"prompts": {
				"bboxes": np.array([[22, 119, 236, 360]]),
				"cls": np.array([0]),
			},
		},
	]

	text_prompts = [
		{
			"image": "ultralytics/assets/bus.jpg",
			"names": ["bus", "man"],
		}
	]

	@classmethod
	def get_visual_prompt(cls, index):
		return cls.visual_prompts[index]

	@classmethod
	def get_text_prompt(cls, index):
		return cls.text_prompts[index]


from ultralytics import YOLO,YOLOE



def val_yoloe26_tp(model_weight,model_yaml=None,**kwargs):
    """
    validate yoloe26 with text prompt or visual prompt.
    Args:
        model_path: str, path to the model weight
        mode: str, "text_prompt" or "visual_prompt"
        end2end: bool, whether to use end2end model, default True. It will achieve better performance when end2end is False.
    Returns:
        model: YOLOE model with validation results. Access model.metrics for DetMetrics, model.val_stats for full COCO eval results
    """
    from ultralytics import YOLOE
    # load model
    if model_yaml:
        model=YOLOE(model_yaml).load(model_weight)
    else:
        model=YOLOE(model_weight)


    #end2end 
    end2end=kwargs.pop("end2end", True)
    if not end2end:
        model.model.end2end = False
        del model.model.model[-1].one2one_cv2
        del model.model.model[-1].one2one_cv3
        del model.model.model[-1].one2one_cv4

    # model.args["clip_weight_name"]=kwargs.pop("clip_weight_name", "mobileclip2:b")

    # arugments
    kwargs["data"]=kwargs.get("data","../datasets/lvis.yaml")
    kwargs["split"]=kwargs.get("split","minival")
    kwargs["max_det"]=kwargs.get("max_det",1000)
    kwargs["save_json"]=kwargs.get("save_json",True)
    kwargs["device"]=kwargs.get("device","0")
    kwargs["conf"]=kwargs.get("conf",0.001)
    model.val(**kwargs)
    return model


# val_yoloe26_tp("yoloe-26l-seg.pt",device="cuda:7")
"""
Evaluating faster-coco-eval mAP using /home/louis/runs/segment/val/predictions.json and ../datasets/lvis/annotations/lvis_v1_minival.json...
Evaluate annotation type *bbox*
COCOeval_opt.evaluate() finished...
DONE (t=17.61s).
Accumulating evaluation results...
COCOeval_opt.accumulate() finished...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.369
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.485
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.397
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.470
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.598
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.364
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.774
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.676
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.578
Evaluate annotation type *segm*
COCOeval_opt.evaluate() finished...
DONE (t=33.04s).
Accumulating evaluation results...
COCOeval_opt.accumulate() finished...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.303
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.455
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.516
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.288
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.268
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.675
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.641
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.473

"""

def val_yoloe_tp(model_weight,model_yaml=None,**kwargs):
    """
    validate yoloe26 with text prompt or visual prompt.
    Args:
        model_path: str, path to the model weight
        mode: str, "text_prompt" or "visual_prompt"
        end2end: bool, whether to use end2end model, default True. It will achieve better performance when end2end is False.
    Returns:
        model: YOLOE model with validation results. Access model.metrics for DetMetrics, model.val_stats for full COCO eval results
    """
    from ultralytics import YOLOE

    # load model
    if model_yaml:
        model=YOLOE(model_yaml).load(model_weight)
    else:
        model=YOLOE(model_weight)



    # model.args["clip_weight_name"]=kwargs.pop("clip_weight_name", "mobileclip2:b")

    # arugments
    kwargs["data"]=kwargs.get("data","../datasets/lvis.yaml")
    kwargs["split"]=kwargs.get("split","minival")
    kwargs["max_det"]=kwargs.get("max_det",1000)
    kwargs["save_json"]=kwargs.get("save_json",True)
    kwargs["device"]=kwargs.get("device","0")
    kwargs["conf"]=kwargs.get("conf",0.001)
    model.val(**kwargs)
    return model

# val_yoloe_tp("yoloe-11l-seg.pt",model_yaml= "yoloe-11l.yaml",device="cuda:7")
'''
Evaluating faster-coco-eval mAP using /home/louis/runs/detect/val/predictions.json and ../datasets/lvis/annotations/lvis_v1_minival.json...
Evaluate annotation type *bbox*
COCOeval_opt.evaluate() finished...
DONE (t=16.77s).
Accumulating evaluation results...
COCOeval_opt.accumulate() finished...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.354
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.466
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.383
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.452
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.596
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.350
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.777
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.665
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.563
'''


def predict_yoloe_tp(model_weight="yoloe-26l-seg.pt", **kwargs):
	"""Text Prompt prediction for YOLOE.
	
	Args:
		model_weight: Path to model weights
		**kwargs: Additional arguments for prediction
			- source: Image path (default: ultralytics/assets/bus.jpg)
			- names: Class names (default: ["bus", "man"])
			- clip_weight_name: CLIP model name
			- save_path: Path to save result image
	
	Returns:
		str: Path to saved result image
	"""
	try:
		kwargs["source"] = kwargs.get("source", "ultralytics/assets/bus.jpg")
		

		model_yaml=kwargs.pop("model_yaml",None)
		if model_yaml:
			model=YOLOE(model_yaml).load(model_weight)
		else:
			model = YOLO(model_weight)

		
		clip_weight_name = kwargs.pop("clip_weight_name", None)
		if clip_weight_name:
			model.args["clip_weight_name"] = clip_weight_name

		names = kwargs.pop("names", ["bus", "man"])
		model.set_classes(names, model.get_text_pe(names))

		save_path = kwargs.pop("save_path", f"./runs/temp/tp_{os.path.basename(model_weight).replace('.pt', '')}_pred.png")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)


		print(f"kwargs: {kwargs}")
		res = model.predict(**kwargs)[0]


		res.save(save_path)

		return save_path
	except Exception as e:
		print(f"❌ Error in predict_yoloe_tp: {e}")
		raise



def predict_yoloe_vp(model_weight="yoloe-26l-seg.pt", **kwargs):
	"""Visual Prompt prediction for YOLOE.
	
	Args:
		model_weight: Path to model weights
		**kwargs: Additional arguments for prediction
			- source: Image path
			- visual_prompts: Visual prompt bboxes and classes
			- predictor: Custom predictor class
			- save_path: Path to save result image
	
	Returns:
		str: Path to saved result image
	"""
	try:
		model_yaml=kwargs.pop("model_yaml",None)
		if model_yaml:
			model=YOLOE(model_yaml).load(model_weight)
		else:
			model = YOLOE(model_weight)

		clip_weight_name = kwargs.pop("clip_weight_name", None)
		if clip_weight_name:
			model.args["clip_weight_name"] = clip_weight_name

		from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

		kwargs["source"] = kwargs.get("source", TestSample.get_visual_prompt(0)["image"])
		kwargs["visual_prompts"] = kwargs.get("visual_prompts", TestSample.get_visual_prompt(0)["prompts"])
		kwargs["predictor"] = kwargs.get("predictor", YOLOEVPSegPredictor)

		save_path = kwargs.pop("save_path", f"./runs/temp/vp_{os.path.basename(model_weight).replace('.pt', '')}_pred.png")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		res.save(save_path)

		return save_path
	except Exception as e:
		print(f"❌ Error in predict_yoloe_vp: {e}")
		raise


import torch
from ultralytics import YOLOE,YOLO
import os

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




def test_yoloe26_tp_main(model_weight="yoloe-26l-seg.pt",model_yaml=None,code_open=True):
	from ultralytics import YOLOE

	if model_yaml:
		model=YOLOE(model_yaml).load(model_weight)
	else:
		model=YOLOE(model_weight)

	model.set_classes(["bus", "person"], model.get_text_pe(["bus", "person"]))

	res=model.predict(source="ultralytics/assets/bus.jpg")[0]

	res_path=f"./runs_temp/123456_tp_pred_{model_weight.split('/')[-1]}.png"

	os.makedirs(os.path.dirname(res_path), exist_ok=True)
	res.save(res_path)

	if code_open:
		from ultra_ext.utils import open_in_vscode
		open_in_vscode(res_path)

def test_yoloe26_vp_main(model_weight="yoloe-26l-seg.pt",model_yaml=None,code_open=True,**kwargs):
	from ultralytics import YOLOE
	from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEVPDetectPredictor
	if model_yaml:
		model=YOLOE(model_yaml).load(model_weight)
	else:
		model=YOLOE(model_weight)
	
	end2end=kwargs.pop("end2end", True)

	if not end2end:
		model.model.end2end=False

	res=model.predict(source=TestSample.get_visual_prompt(0)["image"],
					  visual_prompts=TestSample.get_visual_prompt(0)["prompts"],
                      **kwargs,
					  predictor=YOLOEVPSegPredictor)[0]

	res_path=f"./runs_temp/123456_vp_pred_{model_weight.split('/')[-1]}.png"
	os.makedirs(os.path.dirname(res_path), exist_ok=True)
	res.save(res_path)
	if code_open:
		from ultra_ext.utils import open_in_vscode
		open_in_vscode(res_path)






def yoloev8_flops():

    from ultralytics import YOLO

    for model_weight in ["yoloe-v8s-seg.pt","yoloe-v8m-seg.pt","yoloe-v8l-seg.pt"]:
        for mode in ["tp","vp"]:
            model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
            model_head=model.model.model[-1]
            if mode=="vp":
                del model_head.reprta
            if mode=="tp":
                del model_head.savpe
            model.info()


    # Transferred 448/448 items from pretrained weights
    # YOLOe-v8s summary: 135 layers, 12,912,246 parameters, 12,912,230 gradients, 29.8 GFLOPs
    # Transferred 448/448 items from pretrained weights
    # YOLOe-v8s summary: 163 layers, 13,187,926 parameters, 13,187,910 gradients, 29.8 GFLOPs
    # Transferred 568/568 items from pretrained weights
    # YOLOe-v8m summary: 175 layers, 27,731,270 parameters, 27,731,254 gradients, 80.7 GFLOPs
    # Transferred 568/568 items from pretrained weights
    # YOLOe-v8m summary: 203 layers, 29,751,974 parameters, 29,751,958 gradients, 80.7 GFLOPs
    # Transferred 688/688 items from pretrained weights
    # YOLOe-v8l summary: 215 layers, 45,603,094 parameters, 45,603,078 gradients, 167.6 GFLOPs
    # Transferred 688/688 items from pretrained weights
    # YOLOe-v8l summary: 243 layers, 49,590,006 parameters, 49,589,990 gradients, 167.6 GFLOPs

    # | Model      | Layers      | Params (M)    | GFLOPs          |
    # |------------|-------------|---------------|-----------------|
    # | YOLOE-v8s  | 135 / 163   | 12.3 / 12.6   |  29.8 /  29.8   |
    # | YOLOE-v8m  | 175 / 203   | 26.4 / 28.4   |  80.7 /  80.7   |
    # | YOLOE-v8l  | 215 / 243   | 43.5 / 47.3   | 167.6 / 167.6   |
    # Layers / Params / GFLOPs: TP / VP, GFLOPs measured at 640x640 input

def yoloe11_flops():

    from ultralytics import YOLO

    for model_weight in ["yoloe-11s-seg.pt","yoloe-11m-seg.pt","yoloe-11l-seg.pt"]:
        for mode in ["tp","vp"]:
            model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
            model_head=model.model.model[-1]
            if mode=="vp":
                del model_head.reprta
            if mode=="tp":
                del model_head.savpe
            model.info()


    # Transferred 592/592 items from pretrained weights
    # YOLOe-11s summary: 187 layers, 11,204,438 parameters, 11,204,422 gradients, 22.7 GFLOPs
    # Transferred 592/592 items from pretrained weights
    # YOLOe-11s summary: 215 layers, 11,480,118 parameters, 11,480,102 gradients, 22.7 GFLOPs
    # Transferred 742/742 items from pretrained weights
    # YOLOe-11m summary: 237 layers, 22,026,262 parameters, 22,026,246 gradients, 70.4 GFLOPs
    # Transferred 742/742 items from pretrained weights
    # YOLOe-11m summary: 265 layers, 26,013,174 parameters, 26,013,158 gradients, 70.4 GFLOPs
    # Transferred 1108/1108 items from pretrained weights
    # YOLOe-11l summary: 363 layers, 27,283,734 parameters, 27,283,718 gradients, 89.5 GFLOPs
    # Transferred 1108/1108 items from pretrained weights
    # YOLOe-11l summary: 391 layers, 31,270,646 parameters, 31,270,630 gradients, 89.5 GFLOPs

    # | Model      | Layers      | Params (M)    | GFLOPs        |
    # |------------|-------------|---------------|---------------|
    # | YOLOE-11s  | 187 / 215   | 10.7 / 10.9   | 22.7 / 22.7   |
    # | YOLOE-11m  | 237 / 265   | 21.0 / 24.8   | 70.4 / 70.4   |
    # | YOLOE-11l  | 363 / 391   | 26.0 / 29.8   | 89.5 / 89.5   |
    # Layers / Params / GFLOPs: TP / VP, GFLOPs measured at 640x640 input


def yoloe26_not_e2e_flops():

    from ultralytics import YOLO
    names = ["bus", "man"]


    for model_weight in ["yoloe-26n-seg.pt","yoloe-26s-seg.pt","yoloe-26m-seg.pt", "yoloe-26l-seg.pt","yoloe-26x-seg.pt"]:

        for mode in ["tp","vp"]:
            
            print(f"model_weight: {model_weight}, mode: {mode}")
            model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
            model.set_classes(names, model.get_text_pe(names))
            model_head=model.model.model[-1]
            if mode=="vp":
                del model_head.reprta
            if mode=="tp":
                del model_head.savpe
            del model_head.one2one_cv2
            del model_head.one2one_cv3
            del model_head.one2one_cv4

            model.info()


    # model_weight: yoloe-26n-seg.pt, mode: tp
    # Transferred 202/822 items from pretrained weights
    # YOLOe-26n summary: 221 layers, 4,100,930 parameters, 4,100,930 gradients, 6.1 GFLOPs
    # model_weight: yoloe-26n-seg.pt, mode: vp
    # Transferred 202/822 items from pretrained weights
    # YOLOe-26n summary: 249 layers, 3,223,234 parameters, 3,223,234 gradients, 6.1 GFLOPs
    # model_weight: yoloe-26s-seg.pt, mode: tp
    # Transferred 202/822 items from pretrained weights
    # YOLOe-26s summary: 221 layers, 11,258,578 parameters, 11,258,578 gradients, 21.9 GFLOPs
    # model_weight: yoloe-26s-seg.pt, mode: vp
    # Transferred 202/822 items from pretrained weights
    # YOLOe-26s summary: 249 layers, 11,534,258 parameters, 11,534,258 gradients, 21.9 GFLOPs
    # model_weight: yoloe-26m-seg.pt, mode: tp
    # Transferred 212/882 items from pretrained weights
    # YOLOe-26m summary: 241 layers, 22,346,834 parameters, 22,346,834 gradients, 70.6 GFLOPs
    # model_weight: yoloe-26m-seg.pt, mode: vp
    # Transferred 212/882 items from pretrained weights
    # YOLOe-26m summary: 269 layers, 26,333,746 parameters, 26,333,746 gradients, 70.6 GFLOPs
    # model_weight: yoloe-26l-seg.pt, mode: tp
    # Transferred 266/1206 items from pretrained weights
    # YOLOe-26l summary: 353 layers, 26,750,290 parameters, 26,750,290 gradients, 89.0 GFLOPs
    # model_weight: yoloe-26l-seg.pt, mode: vp
    # Transferred 266/1206 items from pretrained weights
    # YOLOe-26l summary: 381 layers, 30,737,202 parameters, 30,737,202 gradients, 89.0 GFLOPs
    # model_weight: yoloe-26x-seg.pt, mode: tp
    # Transferred 266/1206 items from pretrained weights
    # YOLOe-26x summary: 353 layers, 57,850,354 parameters, 57,850,354 gradients, 197.7 GFLOPs
    # model_weight: yoloe-26x-seg.pt, mode: vp
    # Transferred 266/1206 items from pretrained weights
    # YOLOe-26x summary: 381 layers, 68,399,314 parameters, 68,399,314 gradients, 197.7 GFLOPs


       # | Model      | Layers      | Params (M)    | GFLOPs          |
    # |------------|-------------|---------------|-----------------|
    # | YOLOE-26n  | 221 / 249   |  3.9 /  3.1   |   6.1 /   6.1   |
    # | YOLOE-26s  | 221 / 249   | 10.7 / 11.0   |  21.9 /  21.9   |
    # | YOLOE-26m  | 241 / 269   | 21.3 / 25.1   |  70.6 /  70.6   |
    # | YOLOE-26l  | 353 / 381   | 25.5 / 29.3   |  89.0 /  89.0   |
    # | YOLOE-26x  | 353 / 381   | 55.2 / 65.2   | 197.7 / 197.7   |
    # Layers / Params / GFLOPs: TP / VP, GFLOPs measured at 640x640 input


def yoloe26_seg_not_e2e_flops():

    from ultralytics import YOLOE
    from ultralytics import YOLO

    from ultralytics import YOLO
    names = ["bus", "man"]


    for model_weight in ["yoloe-26n-seg.pt","yoloe-26s-seg.pt","yoloe-26m-seg.pt", "yoloe-26l-seg.pt","yoloe-26x-seg.pt"]:

        for mode in ["tp","vp"]:
            
            print(f"model_weight: {model_weight}, mode: {mode}")
            model=YOLO(model_weight)
            model.set_classes(names, model.get_text_pe(names))
            model_head=model.model.model[-1]
            if mode=="vp":
                del model_head.reprta
            if mode=="tp":
                del model_head.savpe
            del model_head.one2one_cv2
            del model_head.one2one_cv3
            del model_head.one2one_cv4

            model.info()

    # model_weight: yoloe-26n-seg.pt, mode: tp
    # YOLOe-26n-seg summary (fused): 154 layers, 4,645,546 parameters, 1,731,106 gradients, 9.7 GFLOPs
    # model_weight: yoloe-26n-seg.pt, mode: vp
    # YOLOe-26n-seg summary (fused): 172 layers, 3,767,114 parameters, 316,130 gradients, 9.7 GFLOPs
    # model_weight: yoloe-26s-seg.pt, mode: tp
    # YOLOe-26s-seg summary (fused): 154 layers, 12,736,530 parameters, 1,859,362 gradients, 35.2 GFLOPs
    # model_weight: yoloe-26s-seg.pt, mode: vp
    # YOLOe-26s-seg summary (fused): 172 layers, 13,011,042 parameters, 538,850 gradients, 35.2 GFLOPs
    # model_weight: yoloe-26m-seg.pt, mode: tp
    # YOLOe-26m-seg summary (fused): 164 layers, 27,535,938 parameters, 2,269,474 gradients, 123.4 GFLOPs
    # model_weight: yoloe-26m-seg.pt, mode: vp
    # YOLOe-26m-seg summary (fused): 182 layers, 31,520,530 parameters, 1,200,866 gradients, 123.4 GFLOPs
    # model_weight: yoloe-26l-seg.pt, mode: tp
    # YOLOe-26l-seg summary (fused): 222 layers, 31,932,226 parameters, 0 gradients, 141.6 GFLOPs
    # model_weight: yoloe-26l-seg.pt, mode: vp
    # YOLOe-26l-seg summary (fused): 240 layers, 35,916,818 parameters, 0 gradients, 141.6 GFLOPs
    # model_weight: yoloe-26x-seg.pt, mode: tp
    # YOLOe-26x-seg summary (fused): 222 layers, 69,499,970 parameters, 2,810,658 gradients, 316.3 GFLOPs
    # model_weight: yoloe-26x-seg.pt, mode: vp
    # YOLOe-26x-seg summary (fused): 240 layers, 80,045,458 parameters, 1,993,954 gradients, 316.3 GFLOPs

	   # | Model          | Layers      | Params (M)    | GFLOPs          |
    # |----------------|-------------|---------------|-----------------|
    # | YOLOE-26n-seg  | 154 / 172   |  4.4 /  3.6   |   9.7 /   9.7   |
    # | YOLOE-26s-seg  | 154 / 172   | 12.1 / 12.4   |  35.2 /  35.2   |
    # | YOLOE-26m-seg  | 164 / 182   | 26.3 / 30.1   | 123.4 / 123.4   |
    # | YOLOE-26l-seg  | 222 / 240   | 30.5 / 34.3   | 141.6 / 141.6   |
    # | YOLOE-26x-seg  | 222 / 240   | 66.3 / 76.3   | 316.3 / 316.3   |
    # Layers / Params / GFLOPs: TP / VP, GFLOPs measured at 640x640 input



def yoloe26_pf_not_e2e_flops():

    from ultralytics import YOLO
    names = ["bus", "man"]


    for model_weight in ["yoloe-26n-seg-pf.pt","yoloe-26s-seg-pf.pt","yoloe-26m-seg-pf.pt", "yoloe-26l-seg-pf.pt","yoloe-26x-seg-pf.pt"]:

        for mode in ["tp","vp"]:
            
            print(f"model_weight: {model_weight}, mode: {mode}")
            model=YOLO(model_weight)
            model.set_classes(names, model.get_text_pe(names))
            model_head=model.model.model[-1]
            if mode=="vp":
                del model_head.reprta
            if mode=="tp":
                del model_head.savpe
            del model_head.one2one_cv2
            del model_head.one2one_cv3
            del model_head.one2one_cv4

            model.info()


	

def yoloe26_pf_not_e2e_flops():

    from ultralytics import YOLO
    names = ["bus", "man"]


    dirname="./weights/yoloe26_weight/yoloe26_pf"

    for model_weight in ["yoloe26n_pf.pt","yoloe26s_pf.pt","yoloe26m_pf.pt","yoloe26l_pf.pt","yoloe26x_pf.pt"]:

        model_weight=os.path.join(dirname,model_weight)
        # print(f"model_weight: {model_weight}, mode: {mode}")
        model=YOLO(model_weight)
        model_head=model.model.model[-1]
        del model_head.reprta

        del model_head.one2one_cv2 
        del model_head.one2one_cv3
        del model_head.one2one_cv4    
        # model.fuse()
        model.info()


    # YOLOe-26n summary: 219 layers, 2,383,407 parameters, 0 gradients,  5.3 GFLOPs
    # YOLOe-26s summary: 219 layers, 9,482,319 parameters, 0 gradients, 20.8 GFLOPs
    # YOLOe-26m summary: 239 layers, 20,374,351 parameters, 0 gradients, 68.4 GFLOPs
    # YOLOe-26l summary: 351 layers, 24,777,807 parameters, 0 gradients, 86.8 GFLOPs
    # YOLOe-26x summary: 351 layers, 55,681,647 parameters, 0 gradients, 194.4 GFLOPs

    # | Model      | Layers | Params (M) | GFLOPs |
    # |------------|--------|------------|--------|
    # | YOLOE-26n  |  219   |    2.3     |   5.3  |
    # | YOLOE-26s  |  219   |    9.0     |  20.8  |
    # | YOLOE-26m  |  239   |   19.4     |  68.4  |
    # | YOLOE-26l  |  351   |   23.6     |  86.8  |
    # | YOLOE-26x  |  351   |   53.1     | 194.4  |
    # Prompt-free mode (no TP/VP split), GFLOPs measured at 640x640 input


"""



def get_text_feats(model_weight,texts,clip_weight_name="mobileclip2:b",without_reprta=True):
    from ultralytics import YOLOE


    if model_weight is not None:
        model=YOLOE(model_weight)
        model.args['clip_weight_name']=clip_weight_name
        txt_feats=model.model.get_text_pe(texts, without_reprta=without_reprta).squeeze(0)
    else:
        from ultralytics.nn.text_model import OpenCLIP
        import torch
        assert clip_weight_name.endswith("mobileclip2_b.pt"), "当前仅支持 mobileclip2_b.pt 权重文件"
        text_model=OpenCLIP("cuda",pretrained_path=clip_weight_name)
        text_token = text_model.tokenize(texts)
        batch=80
        txt_feats = [text_model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(texts), txt_feats.shape[-1]).squeeze(0)


    print("Text features shape:", txt_feats.shape)

    feat1=txt_feats[0]
    feat2=txt_feats[1]
    print("feat1_norm:", feat1.norm(dim=-1, keepdim=True))
    print("feat2_norm:", feat2.norm(dim=-1, keepdim=True))
    # cal the cosine similarity between feat1 and feat2
    feat1_norm=feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2_norm=feat2 / feat2.norm(dim=-1, keepdim=True)
    cosine_sim=(feat1_norm * feat2_norm).sum(dim=-1)
    # print the norms and cosine similarity

    print("cosine_sim:", cosine_sim)

    return txt_feats

	
"""



def test_resave_yoloe_models_main(model_dir="./weights/yoloe26_weight/yoloe26_vp_seg",do_test=True,wait_input=True):	
    """
    Transfer a model trained with old branch to the new branch by resaving the model weight with the new code. This is to verify the compatibility of the model weight between the old and new code, and to test the performance of the resaved model. The function will resave both the detection model and the segmentation model, and test them if do_test is True. It will wait for user input before processing the next model if wait_input is True.
    Args:
        model_dir: str, the directory of the model weights
        do_test: bool, whether to test the resaved model
        wait_input: bool, whether to wait for user input before processing the next model
    Returns:
        None
    """

    scales=["26n","26s","26m","26l","26x"]

    for scale in scales:

        model_weight=f"{model_dir}/yoloe-{scale}-seg.pt"
        from ultralytics.utils.torch_utils import strip_optimizer

        # resave det model
        strip_optimizer(model_weight)        
        model_yaml=f"yoloe-{scale}.yaml"
        resave_weight=os.path.join(model_dir,f"yoloe-{scale}-resave.pt")
        YOLOE(model_yaml).load(model_weight).save(resave_weight)
        print(f"Resaved {model_weight} to {resave_weight}")

        # test resaved model
        if do_test:
            test_yoloe26_tp_main(model_weight=resave_weight,model_yaml=None,code_open=True)
        # wait user input before processing next model
        if wait_input:
            input(f"Press Enter to continue to the next model...")


        # resave seg model
        seg_model_yaml=f"yoloe-{scale}-seg.yaml"
        resave_seg_weight=os.path.join(model_dir,f"yoloe-{scale}-seg-resave.pt")
        YOLOE(seg_model_yaml).load(model_weight).save(resave_seg_weight)
        print(f"Resaved {model_weight} to {resave_seg_weight}")

        # test resaved model
        if do_test:
            test_yoloe26_tp_main(model_weight=resave_seg_weight,model_yaml=None,code_open=True)
        # wait user input before processing next model
        if wait_input:
            input(f"Press Enter to continue to the next model...")
