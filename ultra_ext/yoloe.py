
import numpy as np

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
		
		model = YOLO(model_weight)
		
		clip_weight_name = kwargs.pop("clip_weight_name", None)
		if clip_weight_name:
			model.args["clip_weight_name"] = clip_weight_name

		names = kwargs.pop("names", ["bus", "man"])
		model.set_classes(names, model.get_text_pe(names))

		print(f"kwargs: {kwargs}")
		res = model.predict(**kwargs)[0]

		save_path = kwargs.get("save_path", f"./runs/temp/tp_{model_weight.replace('.pt', '')}_pred.png")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
		model = YOLO(model_weight)

		clip_weight_name = kwargs.pop("clip_weight_name", None)
		if clip_weight_name:
			model.args["clip_weight_name"] = clip_weight_name

		from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

		kwargs["source"] = kwargs.get("source", TestSample.get_visual_prompt(0)["image"])
		kwargs["visual_prompts"] = kwargs.get("visual_prompts", TestSample.get_visual_prompt(0)["prompts"])
		kwargs["predictor"] = kwargs.get("predictor", YOLOEVPSegPredictor)

		res = model.predict(**kwargs)[0]

		# ✅ 改为 vp，避免与 tp 冲突
		save_path = kwargs.get("save_path", f"./runs/temp/vp_{model_weight.replace('.pt', '')}_pred.png")
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


