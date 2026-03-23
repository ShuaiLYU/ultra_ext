

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


