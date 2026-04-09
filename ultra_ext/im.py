

from pathlib import Path
import numpy as np
import os
import cv2


def _add_title_bar(img, title, font_scale=0.8, font_thickness=2):
    """Prepend a black title bar with centered white text above an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    bar_h = text_size[1] + 16
    bar = np.zeros((bar_h, img.shape[1], img.shape[2]), dtype=img.dtype)
    text_x = max((img.shape[1] - text_size[0]) // 2, 5)
    text_y = bar_h - 6
    cv2.putText(bar, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return np.concatenate([bar, img], axis=0)


def _build_grid(images, max_per_line, padding, pad_value=128):
    """Arrange images into a grid with padding between cells."""
    chunks = [images[i:i + max_per_line] for i in range(0, len(images), max_per_line)]
    ch = images[0].shape[2]

    def make_row(row_imgs):
        max_h = max(img.shape[0] for img in row_imgs)
        cells = []
        for img in row_imgs:
            if img.shape[0] < max_h:
                pad = np.full((max_h - img.shape[0], img.shape[1], ch), pad_value, dtype=img.dtype)
                img = np.concatenate([img, pad], axis=0)
            cells.append(img)
        if padding > 0:
            vsep = np.full((max_h, padding, ch), pad_value, dtype=cells[0].dtype)
            row = cells[0]
            for cell in cells[1:]:
                row = np.concatenate([row, vsep, cell], axis=1)
        else:
            row = np.concatenate(cells, axis=1)
        return row

    rows = [make_row(chunk) for chunk in chunks]
    max_w = max(r.shape[1] for r in rows)

    padded_rows = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], ch), pad_value, dtype=row.dtype)
            row = np.concatenate([row, pad], axis=1)
        padded_rows.append(row)

    if len(padded_rows) == 1:
        return padded_rows[0]

    if padding > 0:
        hsep = np.full((padding, max_w, ch), pad_value, dtype=padded_rows[0].dtype)
        result = padded_rows[0]
        for row in padded_rows[1:]:
            result = np.concatenate([result, hsep, row], axis=0)
    else:
        result = np.concatenate(padded_rows, axis=0)
    return result


def concatenate_images_by_prefix(root, prefix, axis=1, set_title=True, font_scale=0.8, font_thickness=2,
                                  max_img_per_line=None, padding=10):
    """
    Concatenate images that share the same filename across folders matching a prefix.

    Args:
        root (str): Root directory path
        prefix (str): Prefix to match for folder names
        axis (int): Concatenation axis when max_img_per_line is None
                    (0=vertical/top-bottom, 1=horizontal/left-right)
        set_title (bool): Whether to add folder suffix as title bar on each sub-image
        font_scale (float): Font scale for titles (default: 0.8)
        font_thickness (int): Font thickness for titles (default: 2)
        max_img_per_line (int or None): Max sub-images per row in the grid;
                                              if None, all images are placed in a single line
        padding (int): Spacing in pixels between sub-images (default: 10)

    Returns:
        dict: Dictionary with statistics about processed images
    """
    root = Path(root)
    output_dir = root / f"concat_{prefix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all folders matching the prefix
    matching_folders = sorted([
        folder for folder in root.iterdir()
        if folder.is_dir() and folder.name.startswith(prefix)
    ])

    if not matching_folders:
        print(f"No folders found with prefix '{prefix}' in {root}")
        return {"processed": 0, "error": "No folders found"}

    print(f"Found {len(matching_folders)} folders matching prefix '{prefix}':")
    for folder in matching_folders:
        print(f"  - {folder.name}")

    # Collect all files from matching folders
    image_files = {}  # {filename: [(folder_name, filepath), ...]}
    for folder in matching_folders:
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                image_files.setdefault(file.name, []).append((folder.name, file))

    # Process files that appear in multiple folders
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filename, file_list in image_files.items():
        if len(file_list) < 2:
            skipped_count += 1
            continue

        try:
            images = []
            folder_suffixes = []
            for folder_name, filepath in sorted(file_list):
                img = cv2.imread(str(filepath))
                if img is None:
                    print(f"Warning: Failed to load {filepath}")
                    error_count += 1
                    continue
                images.append(img)
                suffix = folder_name[len(prefix):] if folder_name.startswith(prefix) else folder_name
                folder_suffixes.append(suffix)

            if len(images) < 2:
                skipped_count += 1
                continue

            # Optionally stamp a title bar onto each sub-image
            if set_title:
                images = [_add_title_bar(img, suffix, font_scale, font_thickness)
                          for img, suffix in zip(images, folder_suffixes)]

            # Build grid or single-line concatenation
            if max_img_per_line is not None:
                concatenated = _build_grid(images, max_img_per_line, padding)
            else:
                concatenated = _build_grid(images, len(images), padding)

            output_path = output_dir / filename
            cv2.imwrite(str(output_path), concatenated)
            processed_count += 1
            print(f"Saved: {filename} (concatenated {len(images)} images)")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (single copies): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")

    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "output_dir": str(output_dir)
    }


# Example usage:
# result = concatenate_images_by_prefix('/path/to/root', 'my_prefix', max_img_per_line=3, padding=10)




def concat_images_sameh(im_files, output_path):
    images = [cv2.imread(im_file) for im_file in im_files]
    if any(img is None for img in images):
        raise ValueError("One or more images could not be read. Please check the file paths.")
    
    # Resize images to the same height
    min_height = min(img.shape[0] for img in images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
    
    # Concatenate images horizontally
    concatenated_image = cv2.hconcat(resized_images)
    
    # Save the concatenated image
    cv2.imwrite(output_path, concatenated_image)