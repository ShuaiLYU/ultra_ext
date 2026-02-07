"""
RefCOCO Dataset Utilities

This module provides classes for working with RefCOCO referring expression dataset
and downloading COCO images on-demand.
"""

import os
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List

import requests
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class COCO2014:
    """COCO 2014 Dataset class for downloading images from the internet on-demand."""
    
    BASE_URL = "http://images.cocodataset.org"
    
    def __init__(self):
        """Initialize COCO2014 dataset. Images are downloaded from COCO servers on demand."""
        print("Initialized COCO2014 dataset (online mode)")
        print(f"  Base URL: {self.BASE_URL}")
    
    def get_im(self, file_name: str) -> Optional[Image.Image]:
        """
        Download and return image by filename.
        
        Args:
            file_name: COCO image filename (e.g., "COCO_train2014_000000581857.jpg")
                      Can also include RefCOCO suffix like "COCO_train2014_000000581857_16.jpg"
        
        Returns:
            PIL Image object or None if download fails
        """
        match = re.match(r'COCO_(train2014|val2014)_(\d+)', file_name)
        if not match:
            print(f"Error: Invalid COCO filename format: {file_name}")
            return None
        
        split = match.group(1)
        image_id = match.group(2)
        base_name = f"COCO_{split}_{image_id}.jpg"
        url = f"{self.BASE_URL}/{split}/{base_name}"
        
        try:
            print(f"Downloading: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            print(f"  ✓ Downloaded: {image.size} {image.mode}")
            return image
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Download failed: {e}")
            return None
    
    def get_image_url(self, file_name: str) -> Optional[str]:
        """
        Get URL for image file.
        
        Args:
            file_name: COCO image filename
        
        Returns:
            URL string or None if invalid filename
        """
        match = re.match(r'COCO_(train2014|val2014)_(\d+)', file_name)
        if match:
            split = match.group(1)
            image_id = match.group(2)
            base_name = f"COCO_{split}_{image_id}.jpg"
            return f"{self.BASE_URL}/{split}/{base_name}"
        return None


class RefCOCO:
    """RefCOCO Dataset class for referring expression comprehension."""
    
    def __init__(self, split: str = "train", dataset_name: str = "jxu124/refcoco"):
        """
        Initialize RefCOCO dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test', 'testB')
            dataset_name: HuggingFace dataset name (default: 'jxu124/refcoco')
        """
        print(f"Loading RefCOCO dataset: {dataset_name} ({split} split)...")
        self.dataset = load_dataset(dataset_name, split=split)
        self.coco = COCO2014()
        self.split = split
        print(f"  Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)
    
    def get_sample(self, index: int) -> Tuple[Optional[Image.Image], str, List[float], List[str]]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
        
        Returns:
            Tuple of (im, im_name, bbox, captions)
            - im: PIL Image object (None if download fails)
            - im_name: Image filename (str)
            - bbox: Bounding box [x1, y1, x2, y2] (list)
            - captions: List of text descriptions (list of str)
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range [0, {len(self.dataset)-1}]")
        
        sample = self.dataset[index]
        
        # Extract information
        im_name = sample.get('file_name', '')
        bbox = sample.get('bbox', [])
        
        # Extract captions from sentences
        captions = []
        sentences = sample.get('sentences', [])
        if sentences:
            for s in sentences:
                if isinstance(s, dict):
                    text = s.get('raw', s.get('sent', str(s)))
                    captions.append(text)
                else:
                    captions.append(str(s))
        
        # Fallback to captions field if no sentences
        if not captions:
            captions = sample.get('captions', [])
        
        # Download image
        im = self.coco.get_im(im_name)
        
        return im, im_name, bbox, captions
    
    def __getitem__(self, index: int) -> Tuple[Optional[Image.Image], str, List[float], List[str]]:
        """Enable indexing: refcoco[0] returns (im, im_name, bbox, captions)"""
        return self.get_sample(index)


def visualize_sample(im: Image.Image, im_name: str, bbox: List[float], 
                     captions: List[str], sample_idx: int = 0, save_dir: Optional[str] = None):
    """
    Visualize a RefCOCO sample with bounding box and captions.
    
    Args:
        im: PIL Image object
        im_name: Image filename
        bbox: Bounding box [x1, y1, x2, y2]
        captions: List of text descriptions
        sample_idx: Sample index for display
        save_dir: Directory to save the visualization. If provided, saves image and opens in VSCode.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(im)
    
    # Draw bounding box (format: x1, y1, x2, y2)
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Add title with captions
    title = f"Sample {sample_idx}: {im_name}\n"
    title += "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(captions[:3])])
    ax.set_title(title, fontsize=10, loc='left')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        save_path = os.path.join(save_dir, f"refcoco_sample_{sample_idx}_{im_name}")
        
        # Save the figure
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Open with VSCode
        try:
            subprocess.run(['code', save_path], check=False)
            print(f"Opened in VSCode: {save_path}")
        except FileNotFoundError:
            print("VSCode 'code' command not found. Please ensure VSCode is in PATH.")
        except Exception as e:
            print(f"Failed to open in VSCode: {e}")
    else:
        plt.show()


def main():
    """Example usage of RefCOCO dataset."""
    print("\n" + "="*60)
    print("RefCOCO Dataset Demo")
    print("="*60)
    
    # Initialize dataset
    refcoco = RefCOCO(split="train")
    
    # Get a sample
    sample_idx = 10
    print(f"\nGetting sample {sample_idx}...")
    im, im_name, bbox, captions = refcoco[sample_idx]
    
    # Print info
    print(f"\nSample Info:")
    print(f"  Image name: {im_name}")
    print(f"  Image size: {im.size if im else 'None'}")
    print(f"  BBox [x1,y1,x2,y2]: {bbox}")
    print(f"  Captions ({len(captions)}):")
    for i, cap in enumerate(captions, 1):
        print(f"    {i}. {cap}")
    
    # Visualize
    if im:
        visualize_sample(im, im_name, bbox, captions, sample_idx, save_dir="./runs/temp/refcoco_visualizations")


if __name__ == "__main__":
    main()
