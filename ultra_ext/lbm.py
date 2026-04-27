

# read a random bbox with specified category from labelme json file, and return the bbox and category name



def read_one_bbox_from_labelme_json(labelme_json_path, category_name,flag="first"):
    assert flag in ["first","random"], "flag must be 'first' or 'random'"
    import json
    import random

    with open(labelme_json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    category_bboxes = [shape["points"] for shape in shapes if shape["label"] == category_name]

    if not category_bboxes:
        raise ValueError(f"No bounding boxes found for category '{category_name}' in {labelme_json_path}")

    if flag == "random":
        bbox = random.choice(category_bboxes)
    else:  # flag == "first"
        bbox = category_bboxes[0]

    x_min = min(point[0] for point in bbox)
    y_min = min(point[1] for point in bbox)
    x_max = max(point[0] for point in bbox)
    y_max = max(point[1] for point in bbox)

    return [x_min, y_min, x_max, y_max], category_name


import os


def visual_lbm(im_file, json_file, save_img):
	"""Draw LabelMe polygon shapes onto the image and save to save_img.

	Args:
		im_file (str): Source image path.
		json_file (str): LabelMe JSON annotation path.
		save_img (str): Destination path to write the annotated image.

	Returns:
		str: save_img path (None if json_file does not exist).
	"""
	import cv2
	import json
	import numpy as np

	if not os.path.exists(json_file):
		return None

	im = cv2.imread(im_file)
	if im is None:
		return None

	with open(json_file, "r") as f:
		data = json.load(f)

	colors = [
		(0, 255, 0), (0, 0, 255), (255, 0, 0),
		(0, 255, 255), (255, 0, 255), (255, 165, 0),
	]
	label_color = {}

	for shape in data.get("shapes", []):
		label = shape.get("label", "")
		pts = shape.get("points", [])
		shape_type = shape.get("shape_type", "polygon")

		if label not in label_color:
			label_color[label] = colors[len(label_color) % len(colors)]
		color = label_color[label]

		if shape_type == "polygon" and len(pts) >= 3:
			poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
			cv2.polylines(im, [poly], isClosed=True, color=color, thickness=2)
			overlay = im.copy()
			cv2.fillPoly(overlay, [poly], color=color)
			im = cv2.addWeighted(overlay, 0.25, im, 0.75, 0)
		elif shape_type == "rectangle" and len(pts) == 2:
			(x1, y1), (x2, y2) = pts
			cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
		elif shape_type == "circle" and len(pts) == 2:
			cx, cy = int(pts[0][0]), int(pts[0][1])
			r = int(((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2)**0.5)
			cv2.circle(im, (cx, cy), r, color, 2)

		# draw label text
		if pts:
			tx, ty = int(pts[0][0]), int(pts[0][1]) - 5
			cv2.putText(im, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

	os.makedirs(os.path.dirname(os.path.abspath(save_img)), exist_ok=True)
	cv2.imwrite(save_img, im)
	return save_img
