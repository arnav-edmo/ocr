#!/usr/bin/env python3
"""
Add full page images to annotation JSON files.
Extracts pages from PDF and adds them as base64 to the JSON.
"""

import os
import json
import base64
import argparse
from io import BytesIO

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Error: pdf2image and Pillow are required. Install with: pip install pdf2image pillow")
    exit(1)


def pdf_to_base64_images(pdf_path, dpi=200):
    """Convert PDF pages to base64 encoded images"""
    images = convert_from_path(pdf_path, dpi=dpi)
    base64_images = []
    
    for img in images:
        # Convert PIL Image to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        base64_images.append(f"data:image/jpeg;base64,{img_base64}")
    
    return base64_images


def add_page_images_to_json(json_path, pdf_path, output_path=None):
    """Add full page images to annotation JSON"""
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return False
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    # Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract page images from PDF
    print(f"Extracting page images from PDF: {pdf_path}")
    page_images = pdf_to_base64_images(pdf_path)
    print(f"Extracted {len(page_images)} page images")
    
    # Add page images to JSON
    pages = data.get('pages', [])
    if len(pages) != len(page_images):
        print(f"Warning: Page count mismatch. JSON has {len(pages)} pages, PDF has {len(page_images)} pages")
    
    for i, page in enumerate(pages):
        if i < len(page_images):
            page['page_image_base64'] = page_images[i]
            print(f"Added page image to page {i+1}")
        else:
            print(f"Warning: No image available for page {i+1}")
    
    # Save updated JSON
    if output_path is None:
        output_path = json_path.replace('.json', '_with_page_images.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved updated JSON to: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add full page images to annotation JSON files'
    )
    parser.add_argument(
        'json_file',
        help='Path to annotation JSON file'
    )
    parser.add_argument(
        'pdf_file',
        help='Path to source PDF file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file path (default: input_file_with_page_images.json)',
        default=None
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='DPI for page image extraction (default: 200)'
    )
    
    args = parser.parse_args()
    
    add_page_images_to_json(args.json_file, args.pdf_file, args.output)

