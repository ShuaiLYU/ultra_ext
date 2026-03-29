
import numpy as np



# class ModelWeightZoo:

	
# 	weights=[	
# 		"yoloe-26l-seg.pt",
# 		"./runs/yoloe26_tp_ojb365/baseline/weights/best.pt",

# 		]



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


		res = model.predict(**kwargs)[0]
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

def test_yoloe26_vp_main(model_weight="yoloe-26l-seg.pt",model_yaml=None,code_open=True):
	from ultralytics import YOLOE
	from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEVPDetectPredictor
	if model_yaml:
		model=YOLOE(model_yaml).load(model_weight)
	else:
		model=YOLOE(model_weight)
	res=model.predict(source=TestSample.get_visual_prompt(0)["image"],
					  visual_prompts=TestSample.get_visual_prompt(0)["prompts"],
					  predictor=YOLOEVPSegPredictor)[0]

	res_path=f"./runs_temp/123456_vp_pred_{model_weight.split('/')[-1]}.png"
	os.makedirs(os.path.dirname(res_path), exist_ok=True)
	res.save(res_path)
	if code_open:
		from ultra_ext.utils import open_in_vscode
		open_in_vscode(res_path)


def yoloe26_seg_flops():

    from ultralytics import YOLOE
    from ultralytics import YOLO


    model_weight="yoloe-26n-seg.pt"
    model=YOLO(model_weight)
    model.info()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 


    model_weight="yoloe-26s-seg.pt"
    model=YOLO(model_weight)
    model.info()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 
    
    model_weight="yoloe-26m-seg.pt"
    model=YOLO(model_weight)
    model.info()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 

    model_weight="yoloe-26l-seg.pt"
    model=YOLO(model_weight)
    model.info()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 

    model_weight="yoloe-26x-seg.pt"
    model=YOLO(model_weight)
    model.info()
    non_e2e_model=YOLO(model_weight)
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 

    # YOLOe-26n-seg summary (fused): 203 layers, 5,604,156 parameters, 2,019,252 gradients, 11.5 GFLOPs
    # YOLOe-26n-seg summary (fused): 203 layers, 5,604,156 parameters, 2,019,252 gradients, 9.7 GFLOPs
    # YOLOe-26s-seg summary (fused): 203 layers, 15,251,380 parameters, 2,315,892 gradients, 39.0 GFLOPs
    # YOLOe-26s-seg summary (fused): 203 layers, 15,251,380 parameters, 2,315,892 gradients, 35.2 GFLOPs
    # YOLOe-26m-seg summary (fused): 213 layers, 34,889,124 parameters, 3,174,900 gradients, 135.6 GFLOPs
    # YOLOe-26m-seg summary (fused): 213 layers, 34,889,124 parameters, 3,174,900 gradients, 123.4 GFLOPs
    # YOLOe-26l-seg summary (fused): 271 layers, 39,285,412 parameters, 0 gradients, 153.8 GFLOPs
    # YOLOe-26l-seg summary (fused): 271 layers, 39,285,412 parameters, 0 gradients, 141.6 GFLOPs
    # YOLOe-26x-seg summary (fused): 271 layers, 85,337,444 parameters, 4,164,980 gradients, 342.0 GFLOPs
    # YOLOe-26x-seg summary (fused): 271 layers, 85,337,444 parameters, 4,164,980 gradients, 316.3 GFLOPs

  	# ── With segmentation head ────────────────────────────────────────────────
    # | Model           | Layers | Params (M) | Grads (M) | GFLOPs (E2E) | GFLOPs (non-E2E) |
    # |-----------------|--------|------------|-----------|--------------|------------------|
    # | YOLOE-26n-seg   |  203   |    5.60    |   2.02    |    11.5      |       9.7        |
    # | YOLOE-26s-seg   |  203   |   15.25    |   2.32    |    39.0      |      35.2        |
    # | YOLOE-26m-seg   |  213   |   34.89    |   3.17    |   135.6      |     123.4        |
    # | YOLOE-26l-seg   |  271   |   39.29    |   0.00    |   153.8      |     141.6        |
    # | YOLOE-26x-seg   |  271   |   85.34    |   4.16    |   342.0      |     316.3        |


def yoloe26_flops():

    from ultralytics import YOLO
    names = ["bus", "man"]

    model_weight="yoloe-26n-seg.pt"
    model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)

    model.set_classes(names, model.get_text_pe(names))
    model.info()
    non_e2e_model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    non_e2e_model.set_classes(names, model.get_text_pe(names))
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 


    model_weight="yoloe-26s-seg.pt"
    model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    model.set_classes(names, model.get_text_pe(names))
    model.info()
    non_e2e_model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    non_e2e_model.set_classes(names, model.get_text_pe(names))
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 
    
    model_weight="yoloe-26m-seg.pt"
    model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    model.set_classes(names, model.get_text_pe(names))
    model.info()
    non_e2e_model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    non_e2e_model.set_classes(names, model.get_text_pe(names))
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 

    model_weight="yoloe-26l-seg.pt"
    model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    model.set_classes(names, model.get_text_pe(names))
    model.info()
    non_e2e_model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    non_e2e_model.set_classes(names, model.get_text_pe(names))
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 

    model_weight="yoloe-26x-seg.pt"
    model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    model.set_classes(names, model.get_text_pe(names))
    model.info()
    non_e2e_model=YOLO(model_weight.replace("-seg.pt",".yaml")).load(model_weight)
    non_e2e_model.set_classes(names, model.get_text_pe(names))
    non_e2e_model.model.end2end=False
    non_e2e_model.info() 
	
    # YOLOe-26n summary: 304 layers, 5,067,696 parameters, 5,067,696 gradients, 7.3 GFLOPs
    # Transferred 202/864 items from pretrained weights
    # YOLOe-26n summary: 304 layers, 5,067,696 parameters, 5,067,696 gradients, 6.1 GFLOPs
    # Transferred 202/864 items from pretrained weights
    # YOLOe-26s summary: 304 layers, 13,782,992 parameters, 13,782,992 gradients, 24.8 GFLOPs
    # Transferred 202/864 items from pretrained weights
    # YOLOe-26s summary: 304 layers, 13,782,992 parameters, 13,782,992 gradients, 21.9 GFLOPs
    # Transferred 212/924 items from pretrained weights
    # YOLOe-26m summary: 324 layers, 29,712,464 parameters, 29,712,464 gradients, 79.2 GFLOPs
    # Transferred 212/924 items from pretrained weights
    # YOLOe-26m summary: 324 layers, 29,712,464 parameters, 29,712,464 gradients, 70.6 GFLOPs
    # Transferred 266/1248 items from pretrained weights
    # YOLOe-26l summary: 436 layers, 34,115,920 parameters, 34,115,920 gradients, 97.6 GFLOPs
    # Transferred 266/1248 items from pretrained weights
    # YOLOe-26l summary: 436 layers, 34,115,920 parameters, 34,115,920 gradients, 89.0 GFLOPs
    # Transferred 266/1248 items from pretrained weights
    # YOLOe-26x summary: 436 layers, 73,703,408 parameters, 73,703,408 gradients, 215.2 GFLOPs
    # Transferred 266/1248 items from pretrained weights
    # YOLOe-26x summary: 436 layers, 73,703,408 parameters, 73,703,408 gradients, 197.7 GFLOPs

	# ── Detection-only (no seg head) ──────────────────────────────────────────
    # | Model       | Layers | Params (M) | Grads (M) | GFLOPs (E2E) | GFLOPs (non-E2E) |
    # |-------------|--------|------------|-----------|--------------|------------------|
    # | YOLOE-26n   |  304   |    5.07    |   5.07    |     7.3      |       6.1        |
    # | YOLOE-26s   |  304   |   13.78    |  13.78    |    24.8      |      21.9        |
    # | YOLOE-26m   |  324   |   29.71    |  29.71    |    79.2      |      70.6        |
    # | YOLOE-26l   |  436   |   34.12    |  34.12    |    97.6      |      89.0        |
    # | YOLOE-26x   |  436   |   73.70    |  73.70    |   215.2      |     197.7        |