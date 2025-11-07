#!/usr/bin/env python3
"""
Standalone script to extract images from existing annotation JSON files.
Useful if you already have processed JSON files and want to extract images retroactively.
"""

import os
import json
import base64
import re
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_images_from_json(json_path: str, output_dir: str = None):
    """
    Extract base64 images from an annotation JSON file and save them.
    
    Args:
        json_path: Path to the annotation JSON file
        output_dir: Directory to save images (default: same as JSON file directory)
    """
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return
    
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine output directory
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        json_dir = os.path.dirname(json_path)
        output_dir = os.path.join(json_dir, f"{base_name}_images")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving images to: {output_dir}")
    
    saved_images = []
    image_counter = 0
    
    # Extract from bbox_annotations
    bbox_annotations = data.get("bbox_annotations", [])
    logger.info(f"Found {len(bbox_annotations)} bbox annotations")
    
    for idx, bbox_ann in enumerate(bbox_annotations):
        if not isinstance(bbox_ann, dict):
            continue
        
        image_base64 = bbox_ann.get('image_base64')
        if not image_base64:
            continue
        
        try:
            # Handle data URI format
            if isinstance(image_base64, str) and image_base64.startswith('data:'):
                match = re.match(r'data:image/(\w+);base64,(.+)', image_base64)
                if match:
                    image_format = match.group(1)
                    base64_data = match.group(2)
                else:
                    parts = image_base64.split(',', 1)
                    base64_data = parts[1] if len(parts) == 2 else image_base64
                    image_format = 'jpeg'
            else:
                base64_data = image_base64
                image_format = 'jpeg'
            
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            
            # Determine file extension
            ext = image_format if image_format in ['jpeg', 'png', 'gif', 'webp'] else 'jpg'
            if ext == 'jpeg':
                ext = 'jpg'
            
            # Generate filename
            image_counter += 1
            filename = f"img_{image_counter:03d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            # Update annotation with file path
            bbox_ann['image_file_path'] = filepath
            bbox_ann['image_filename'] = filename
            
            saved_images.append({
                'index': idx,
                'filename': filename,
                'filepath': filepath,
                'size_bytes': len(image_bytes)
            })
            
            logger.info(f"Saved image {image_counter}: {filename} ({len(image_bytes) / 1024:.1f} KB)")
            
        except Exception as e:
            logger.warning(f"Failed to extract/save image from annotation {idx}: {e}")
    
    # Also check pages for images (could be in 'images' or 'image_annotations')
    pages = data.get("pages", [])
    logger.info(f"Checking {len(pages)} pages for image annotations")
    
    for page_idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        
        # Check both 'images' and 'image_annotations' fields
        image_anns = page.get('images', [])
        if not image_anns:
            image_anns = page.get('image_annotations', [])
        for img_ann in image_anns:
            if not isinstance(img_ann, dict):
                continue
            
            image_base64 = img_ann.get('image_base64')
            if not image_base64:
                continue
            
            try:
                # Same extraction logic
                if isinstance(image_base64, str) and image_base64.startswith('data:'):
                    match = re.match(r'data:image/(\w+);base64,(.+)', image_base64)
                    if match:
                        image_format = match.group(1)
                        base64_data = match.group(2)
                    else:
                        parts = image_base64.split(',', 1)
                        base64_data = parts[1] if len(parts) == 2 else image_base64
                        image_format = 'jpeg'
                else:
                    base64_data = image_base64
                    image_format = 'jpeg'
                
                image_bytes = base64.b64decode(base64_data)
                ext = image_format if image_format in ['jpeg', 'png', 'gif', 'webp'] else 'jpg'
                if ext == 'jpeg':
                    ext = 'jpg'
                
                image_counter += 1
                filename = f"img_{image_counter:03d}.{ext}"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                img_ann['image_file_path'] = filepath
                img_ann['image_filename'] = filename
                
                saved_images.append({
                    'index': f"page_{page_idx}",
                    'filename': filename,
                    'filepath': filepath,
                    'size_bytes': len(image_bytes)
                })
                
                logger.info(f"Saved image {image_counter} from page {page_idx}: {filename} ({len(image_bytes) / 1024:.1f} KB)")
                
            except Exception as e:
                logger.warning(f"Failed to extract/save image from page {page_idx}: {e}")
    
    # Save updated JSON with image paths
    output_json_path = json_path.replace('.json', '_with_images.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Extracted and saved {len(saved_images)} images to {output_dir}")
    logger.info(f"✓ Updated JSON saved to {output_json_path}")
    
    return saved_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract images from annotation JSON files'
    )
    parser.add_argument(
        'json_file',
        help='Path to annotation JSON file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for images (default: JSON_file_images/)',
        default=None
    )
    
    args = parser.parse_args()
    
    extract_images_from_json(args.json_file, args.output_dir)

