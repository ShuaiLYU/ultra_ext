# Ultra Extensions

Extended utilities for YOLO and computer vision models, including datasets and utilities for referring expression comprehension tasks.

## Features

- **RefCOCO Dataset**: Easy-to-use interface for RefCOCO referring expression dataset
- **COCO2014**: On-demand COCO image downloading from official servers
- **Visualization Tools**: Built-in visualization for bounding boxes and captions

## Installation

### From source

```bash
cd /path/to/ultra_louis_work
pip install -e ultra_ext
```

### Standard installation

```bash
pip install ultra_ext
```

## Quick Start

### Using RefCOCO Dataset

```python
from ultra_ext.datasets import RefCOCO

# Initialize dataset
refcoco = RefCOCO(split="train")

# Get a sample
im, im_name, bbox, captions = refcoco[0]

print(f"Image: {im.size}")
print(f"Filename: {im_name}")
print(f"BBox [x1,y1,x2,y2]: {bbox}")
print(f"Captions: {captions}")
```

### Visualizing Samples

```python
from ultra_ext.datasets import RefCOCO, visualize_sample

refcoco = RefCOCO(split="train")
im, im_name, bbox, captions = refcoco[0]

# Visualize with bounding box and captions
if im:
    visualize_sample(im, im_name, bbox, captions, sample_idx=0)
```

### Using COCO2014 Directly

```python
from ultra_ext.datasets import COCO2014

coco = COCO2014()

# Download an image by filename
image = coco.get_im("COCO_train2014_000000581857.jpg")

# Get image URL
url = coco.get_image_url("COCO_train2014_000000581857.jpg")
```

## API Reference

### RefCOCO

**Class: `RefCOCO(split="train", dataset_name="jxu124/refcoco")`**

- `split`: Dataset split ('train', 'validation', 'test', 'testB')
- `dataset_name`: HuggingFace dataset name

**Methods:**
- `get_sample(index)`: Returns `(im, im_name, bbox, captions)`
- `__getitem__(index)`: Enable indexing `refcoco[0]`
- `__len__()`: Returns dataset size

**Returns:**
- `im`: PIL Image object (None if download fails)
- `im_name`: Image filename (str)
- `bbox`: Bounding box [x1, y1, x2, y2] (list)
- `captions`: List of text descriptions (list of str)

### COCO2014

**Class: `COCO2014()`**

**Methods:**
- `get_im(file_name)`: Download and return PIL Image
- `get_image_url(file_name)`: Get URL for image file

### visualize_sample

**Function: `visualize_sample(im, im_name, bbox, captions, sample_idx=0)`**

Visualize a RefCOCO sample with bounding box and captions using matplotlib.

## Requirements

- Python >= 3.8
- PIL/Pillow
- requests
- datasets (HuggingFace)
- matplotlib

## License

MIT License

## Author

Louis
