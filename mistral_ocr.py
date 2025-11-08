#!/usr/bin/env python3
"""
mistral_ocr.py
OCR pipeline using Mistral Document AI:
PDF -> Mistral OCR API -> structured JSON output

Requires:
- MISTRAL_API_KEY environment variable or .env file
- mistralai package installed
"""

import os
import json
import uuid
import logging
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:
    raise ImportError("Please install mistralai: pip install mistralai")

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Please install pydantic: pip install pydantic")


from dotenv import load_dotenv
load_dotenv()

# Optional imports for page image extraction
try:
    from pdf2image import convert_from_path
    from PIL import Image
    from io import BytesIO
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Page images will not be extracted. Install with: pip install pdf2image pillow")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- CONFIG ----
OUTPUT_DIR = "output"

# Initialize Mistral client (will be set in main or from env)
client = None


# ---- Pydantic Models for Annotations ----
# Based on actual Janes Fighting Ships page structure

class IndividualShip(BaseModel):
    """Individual ship entry with name, pennant number, and dates"""
    name: str = Field(..., description="Ship name (e.g., 'ORIKU', 'ILIRIA')")
    pennant_number: Optional[str] = Field(None, description="Pennant number (e.g., 'P 132', 'P 131')")
    ex_number: Optional[str] = Field(None, description="Previous pennant number if applicable (e.g., 'ex-R 216')")
    launch_date: Optional[str] = Field(None, description="Launch date if mentioned")
    commission_date: Optional[str] = Field(None, description="Commission date if mentioned")


class ShipSpecifications(BaseModel):
    """Complete ship specifications for a ship class"""
    class_name: str = Field(..., description="Ship class name (e.g., 'DAMEN STAN PATROL 4207 CLASS')")
    class_type: Optional[str] = Field(None, description="Class type abbreviation (e.g., 'PB', 'PBF', 'FFG')")
    ships: List[IndividualShip] = Field(default_factory=list, description="List of individual ships in this class")
    
    # Specifications
    displacement_tonnes: Optional[float] = Field(None, description="Displacement in tonnes")
    displacement_type: Optional[str] = Field(None, description="Displacement type (e.g., 'standard', 'full load')")
    dimensions_metres: Optional[str] = Field(None, description="Dimensions in metres format (e.g., '42.8 × 7.11 x 2.52')")
    dimensions_feet: Optional[str] = Field(None, description="Dimensions in feet format (e.g., '140.4 x 23.3 x 8.3')")
    speed_knots: Optional[float] = Field(None, description="Maximum speed in knots")
    range_n_miles: Optional[float] = Field(None, description="Range in nautical miles")
    range_at_speed: Optional[str] = Field(None, description="Speed at which range is measured (e.g., 'at 12 kt', 'at 15 kt')")
    complement: Optional[int] = Field(None, description="Number of crew members")
    machinery: Optional[str] = Field(None, description="Machinery description (engines, horsepower, shafts, etc.)")
    guns: Optional[str] = Field(None, description="Guns/armament description")
    radars: Optional[str] = Field(None, description="Radar systems description")
    comment: Optional[str] = Field(None, description="Commentary section about the ship class")


class CountryOverview(BaseModel):
    """Country overview section"""
    country_name: Optional[str] = Field(None, description="Country name")
    political_info: Optional[str] = Field(None, description="Political information")
    geographical_info: Optional[str] = Field(None, description="Geographical information")
    area: Optional[str] = Field(None, description="Area (e.g., '11,100 square miles')")
    coastline: Optional[str] = Field(None, description="Coastline length (e.g., '195 n miles')")
    principal_ports: Optional[List[str]] = Field(default_factory=list, description="Principal ports")
    capital: Optional[str] = Field(None, description="Capital city")


class ForceDetails(BaseModel):
    """Naval force details"""
    force_name: Optional[str] = Field(None, description="Name of the naval force")
    territorial_waters: Optional[str] = Field(None, description="Territorial waters claim")
    personnel: Optional[str] = Field(None, description="Personnel count and year (e.g., '2020: 1,900')")
    commander: Optional[str] = Field(None, description="Commander name and rank")
    bases: Optional[List[str]] = Field(default_factory=list, description="List of bases")


class PageAnnotation(BaseModel):
    """Complete page annotation matching Janes Fighting Ships structure"""
    country: Optional[str] = Field(None, description="Country or region this page covers")
    section_type: Optional[str] = Field(None, description="Section type (e.g., 'Patrol Forces', 'Frigates', 'Destroyers')")
    country_overview: Optional[CountryOverview] = Field(None, description="Country overview section")
    force_details: Optional[ForceDetails] = Field(None, description="Naval force details")
    ship_classes: List[ShipSpecifications] = Field(default_factory=list, description="List of ship classes on this page")


class FigureAnnotation(BaseModel):
    """BBox annotation for figures/images"""
    image_type: str = Field(..., description="Type of image (e.g., 'ship photograph', 'ship diagram', 'chart')")
    caption: Optional[str] = Field(None, description="Image caption if present")
    ship_name: Optional[str] = Field(None, description="Ship name visible in image (e.g., 'ORIKU', 'P 110')")
    pennant_number: Optional[str] = Field(None, description="Pennant number visible in image")
    description: str = Field(..., description="Detailed description of what the image shows")
    credit: Optional[str] = Field(None, description="Photo credit if mentioned")


# ---- Utilities ----
def ensure_dir(d):
    """Ensure directory exists"""
    os.makedirs(d, exist_ok=True)


def extract_and_save_images(annotations_data: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """
    Extract base64 images from annotations (keep only base64, no file saving).
    
    Args:
        annotations_data: Dictionary containing bbox_annotations with image_base64 fields
        pdf_path: Original PDF path (unused, kept for compatibility)
        
    Returns:
        Updated annotations_data (base64 images are already in place)
    """
    # Count images for logging
    image_count = 0
    
    # Count images from bbox_annotations
    bbox_annotations = annotations_data.get("bbox_annotations", [])
    for bbox_ann in bbox_annotations:
        if isinstance(bbox_ann, dict):
            if bbox_ann.get('image_base64'):
                image_count += 1
        elif hasattr(bbox_ann, 'image_base64') and bbox_ann.image_base64:
            image_count += 1
    
    # Count images from pages
    pages = annotations_data.get("pages", [])
    for page in pages:
        image_anns = []
        if isinstance(page, dict):
            image_anns = page.get('images', []) or page.get('image_annotations', [])
        elif hasattr(page, 'images'):
            image_anns = page.images
        elif hasattr(page, 'image_annotations'):
            image_anns = page.image_annotations
        
        for img_ann in image_anns:
            if isinstance(img_ann, dict) and img_ann.get('image_base64'):
                image_count += 1
            elif hasattr(img_ann, 'image_base64') and img_ann.image_base64:
                image_count += 1
    
    logger.info(f"Found {image_count} images with base64 data (kept in JSON, no files saved)")
    
    # Remove any existing file path references
    for bbox_ann in bbox_annotations:
        if isinstance(bbox_ann, dict):
            bbox_ann.pop('image_file_path', None)
            bbox_ann.pop('image_filename', None)
    
    for page in pages:
        image_anns = []
        if isinstance(page, dict):
            image_anns = page.get('images', []) or page.get('image_annotations', [])
        elif hasattr(page, 'images'):
            image_anns = page.images
        elif hasattr(page, 'image_annotations'):
            image_anns = page.image_annotations
        
        for img_ann in image_anns:
            if isinstance(img_ann, dict):
                img_ann.pop('image_file_path', None)
                img_ann.pop('image_filename', None)
    
    # Don't add extracted_images or images_directory - we only use base64
    return annotations_data


def add_page_images_to_json(json_path: str, pdf_path: str) -> bool:
    """
    Add full page images to annotation JSON file.
    Extracts pages from PDF and adds them as base64 to the JSON.
    
    Args:
        json_path: Path to annotation JSON file
        pdf_path: Path to source PDF file
        
    Returns:
        True if successful, False otherwise
    """
    if not PDF2IMAGE_AVAILABLE:
        logger.warning("pdf2image not available. Skipping page image extraction.")
        return False
    
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return False
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    try:
        # Read JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract page images from PDF
        logger.info(f"Extracting page images from PDF: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=200)
        base64_images = []
        
        for img in images:
            # Convert PIL Image to base64
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(f"data:image/jpeg;base64,{img_base64}")
        
        logger.info(f"Extracted {len(base64_images)} page images")
        
        # Add page images to JSON
        pages = data.get('pages', [])
        if len(pages) != len(base64_images):
            logger.warning(f"Page count mismatch. JSON has {len(pages)} pages, PDF has {len(base64_images)} pages")
        
        for i, page in enumerate(pages):
            if i < len(base64_images):
                page['page_image_base64'] = base64_images[i]
                logger.info(f"Added page image to page {i+1}")
            else:
                logger.warning(f"No image available for page {i+1}")
        
        # Save updated JSON (overwrite original)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated JSON with page images: {json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add page images: {e}")
        return False


def generate_viewer_html(json_filename: str, output_dir: str = OUTPUT_DIR) -> str:
    """
    Generate viewer.html file with the correct JSON filename.
    
    Args:
        json_filename: Name of the JSON file (e.g., 'document_annotations.json')
        output_dir: Directory where viewer.html will be saved
        
    Returns:
        Path to generated viewer.html file
    """
    # Load template from file
    template_path = Path(__file__).parent / 'viewer_template.txt'
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        viewer_template = f.read()
    
    # Replace placeholder with actual JSON filename
    viewer_html = viewer_template.replace('{json_filename}', json_filename)
    
    # Save viewer.html
    ensure_dir(output_dir)
    viewer_path = os.path.join(output_dir, 'viewer.html')
    
    with open(viewer_path, 'w', encoding='utf-8') as f:
        f.write(viewer_html)
    
    logger.info(f"Generated viewer.html at {viewer_path}")
    return viewer_path


def process_pdf_with_mistral(pdf_path: str, mistral_client: Optional[Mistral] = None) -> Dict[str, Any]:
    """
    Process PDF using Mistral Document AI OCR
    
    Args:
        pdf_path: Path to the PDF file
        mistral_client: Mistral client instance (uses global client if None)
        
    Returns:
        Dictionary containing OCR results
    """
    if mistral_client is None:
        mistral_client = client
        if mistral_client is None:
            raise ValueError("Mistral client not initialized. Please set MISTRAL_API_KEY or provide --api-key")
    
    logger.info(f"Processing PDF with Mistral Document AI: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Read PDF file as bytes
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    
    logger.info(f"PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
    
    # Step 1: Upload PDF to Mistral's cloud storage
    try:
        logger.info("Uploading PDF to Mistral cloud storage...")
        uploaded_pdf = mistral_client.files.upload(
            file={
                "file_name": os.path.basename(pdf_path),
                "content": pdf_bytes,
            },
            purpose="ocr"
        )
        logger.info(f"PDF uploaded successfully. File ID: {uploaded_pdf.id}")
        
        # Step 2: Process with Mistral OCR using file_id directly with annotations
        logger.info("Sending PDF to Mistral OCR API with annotations...")
        result = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "file",
                "file_id": uploaded_pdf.id,
            },
            document_annotation_format=response_format_from_pydantic_model(PageAnnotation),
            bbox_annotation_format=response_format_from_pydantic_model(FigureAnnotation),
            include_image_base64=True  # Include images in response
        )
        logger.info("OCR processing completed successfully")
        
        # Clean up: delete uploaded file (optional)
        try:
            mistral_client.files.delete(file_id=uploaded_pdf.id)
            logger.info("Cleaned up uploaded file")
        except Exception as e:
            logger.warning(f"Could not delete uploaded file: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error processing PDF with Mistral OCR: {e}")
        raise


def extract_annotations(mistral_result: Any) -> Dict[str, Any]:
    """
    Extract annotations from Mistral result and convert to usable format
    
    Returns:
        Dictionary with document_annotation and bbox_annotations
    """
    result = {
        "document_annotation": None,
        "bbox_annotations": [],
        "text": "",
        "pages": []
    }
    
    # Extract text
    if hasattr(mistral_result, 'text'):
        result["text"] = mistral_result.text
    elif isinstance(mistral_result, dict):
        result["text"] = mistral_result.get('text', '')
    
    # Extract document annotation
    if hasattr(mistral_result, 'document_annotation'):
        doc_ann = mistral_result.document_annotation
        if hasattr(doc_ann, 'model_dump'):
            result["document_annotation"] = doc_ann.model_dump()
        elif hasattr(doc_ann, '__dict__'):
            result["document_annotation"] = doc_ann.__dict__
        else:
            result["document_annotation"] = doc_ann
    elif isinstance(mistral_result, dict):
        result["document_annotation"] = mistral_result.get('document_annotation')
    
    # Extract pages first (needed for bbox annotation extraction)
    pages_data = []
    if hasattr(mistral_result, 'pages'):
        pages = mistral_result.pages
        for page in pages:
            # Handle JSON string pages
            if isinstance(page, str):
                try:
                    page = json.loads(page)
                except:
                    pass
            
            if hasattr(page, 'model_dump'):
                pages_data.append(page.model_dump())
            elif hasattr(page, '__dict__'):
                pages_data.append(page.__dict__)
            elif isinstance(page, dict):
                pages_data.append(page)
            else:
                pages_data.append(page)
    elif isinstance(mistral_result, dict):
        raw_pages = mistral_result.get('pages', [])
        for page in raw_pages:
            # Handle JSON string pages
            if isinstance(page, str):
                try:
                    page = json.loads(page)
                except:
                    pass
            pages_data.append(page)
    result["pages"] = pages_data
    
    # Extract bbox annotations
    # Bbox annotations can be at top level OR nested in pages as image_annotations
    bbox_anns = None
    
    # Try top level first
    if hasattr(mistral_result, 'bbox_annotations'):
        bbox_anns = mistral_result.bbox_annotations
    elif isinstance(mistral_result, dict):
        bbox_anns = mistral_result.get('bbox_annotations')
    
    # If not at top level, extract from pages
    if not bbox_anns:
        for page in pages_data:
            # Handle both dict and object pages
            # Check both 'images' and 'image_annotations' fields
            image_anns = []
            if isinstance(page, dict):
                image_anns = page.get('images', [])
                if not image_anns:
                    image_anns = page.get('image_annotations', [])
            elif hasattr(page, 'images'):
                image_anns = page.images
            elif hasattr(page, 'image_annotations'):
                image_anns = page.image_annotations
            else:
                continue
            
            if image_anns:
                for img_ann in image_anns:
                    # Extract the actual annotation (could be nested)
                    if isinstance(img_ann, dict):
                        # Could be {"image_base64": "...", "image_annotation": "..."}
                        ann_data = img_ann.get('image_annotation')
                        image_base64 = img_ann.get('image_base64')
                        
                        if ann_data:
                            # If it's a JSON string, parse it
                            if isinstance(ann_data, str):
                                try:
                                    ann_data = json.loads(ann_data)
                                except:
                                    pass
                            
                            # If ann_data is a dict, preserve image_base64 if it exists
                            if isinstance(ann_data, dict) and image_base64:
                                ann_data['image_base64'] = image_base64
                            
                            result["bbox_annotations"].append(ann_data)
                        else:
                            # Use the whole object (which should include image_base64)
                            result["bbox_annotations"].append(img_ann)
                    elif hasattr(img_ann, 'model_dump'):
                        result["bbox_annotations"].append(img_ann.model_dump())
                    elif hasattr(img_ann, '__dict__'):
                        result["bbox_annotations"].append(img_ann.__dict__)
                    else:
                        result["bbox_annotations"].append(img_ann)
    
    # Process top-level bbox annotations if found
    if bbox_anns:
        if isinstance(bbox_anns, list):
            for bbox in bbox_anns:
                if hasattr(bbox, 'model_dump'):
                    result["bbox_annotations"].append(bbox.model_dump())
                elif hasattr(bbox, '__dict__'):
                    result["bbox_annotations"].append(bbox.__dict__)
                else:
                    result["bbox_annotations"].append(bbox)
        else:
            if hasattr(bbox_anns, 'model_dump'):
                result["bbox_annotations"].append(bbox_anns.model_dump())
            elif hasattr(bbox_anns, '__dict__'):
                result["bbox_annotations"].append(bbox_anns.__dict__)
            else:
                result["bbox_annotations"].append(bbox_anns)
    
    return result


def extract_annotations_from_result(mistral_result: Any) -> tuple:
    """
    Extract document and bbox annotations from Mistral result
    
    Returns:
        Tuple of (document_annotation, bbox_annotations)
    """
    doc_annotation = None
    bbox_annotations = []
    
    # Try to extract annotations
    if hasattr(mistral_result, 'document_annotation'):
        doc_annotation = mistral_result.document_annotation
    elif hasattr(mistral_result, 'document_annotations'):
        doc_annotation = mistral_result.document_annotations
    
    if hasattr(mistral_result, 'bbox_annotations'):
        bbox_annotations = mistral_result.bbox_annotations
    elif hasattr(mistral_result, 'bbox_annotation'):
        if isinstance(mistral_result.bbox_annotation, list):
            bbox_annotations = mistral_result.bbox_annotation
        else:
            bbox_annotations = [mistral_result.bbox_annotation]
    
    # Try to convert to dict if needed
    if doc_annotation and not isinstance(doc_annotation, dict):
        if hasattr(doc_annotation, 'model_dump'):
            doc_annotation = doc_annotation.model_dump()
        elif hasattr(doc_annotation, '__dict__'):
            doc_annotation = doc_annotation.__dict__
    
    if bbox_annotations:
        converted_bbox = []
        for bbox in bbox_annotations:
            if not isinstance(bbox, dict):
                if hasattr(bbox, 'model_dump'):
                    converted_bbox.append(bbox.model_dump())
                elif hasattr(bbox, '__dict__'):
                    converted_bbox.append(bbox.__dict__)
                else:
                    converted_bbox.append(bbox)
            else:
                converted_bbox.append(bbox)
        bbox_annotations = converted_bbox
    
    return doc_annotation, bbox_annotations


def convert_ship_specs_to_key_values(ship: Dict[str, Any], page_no: int) -> List[Dict[str, Any]]:
    """Convert ship specifications to key_values format"""
    kvs = []
    kv_id_base = f"p{page_no}_kv"
    
    # Displacement
    if ship.get('displacement_tonnes'):
        kvs.append({
            "kv_id": f"{kv_id_base}_disp",
            "field_name": "Displacement_tonnes",
            "raw_text": f"Displacement, tonnes: {ship.get('displacement_tonnes')} {ship.get('displacement_qualifier', '')}".strip(),
            "value_normalized": ship.get('displacement_tonnes'),
            "units": "tonnes",
            "qualifier": ship.get('displacement_qualifier'),
            "page_no": page_no,
            "bbox": [0, 0, 0, 0],  # Will be updated if we have bbox info
            "confidence": 0.95,
            "source": "annotation"
        })
    
    # Dimensions
    dims = ship.get('dimensions')
    if dims:
        if isinstance(dims, dict):
            dim_values = []
            if dims.get('length'):
                dim_values.append(dims['length'])
            if dims.get('beam'):
                dim_values.append(dims['beam'])
            if dims.get('draught'):
                dim_values.append(dims['draught'])
            
            if dim_values:
                kvs.append({
                    "kv_id": f"{kv_id_base}_dims",
                    "field_name": "Dimensions_metres",
                    "raw_text": f"Dimensions: {dim_values[0] if len(dim_values) > 0 else ''} x {dim_values[1] if len(dim_values) > 1 else ''} x {dim_values[2] if len(dim_values) > 2 else ''}",
                    "value_normalized": dim_values,
                    "units": "metres",
                    "page_no": page_no,
                    "bbox": [0, 0, 0, 0],
                    "confidence": 0.95,
                    "source": "annotation"
                })
    
    # Speed
    if ship.get('speed_knots'):
        kvs.append({
            "kv_id": f"{kv_id_base}_speed",
            "field_name": "Speed_knots",
            "raw_text": f"Speed, knots: {ship.get('speed_knots')}",
            "value_normalized": ship.get('speed_knots'),
            "units": "knots",
            "page_no": page_no,
            "bbox": [0, 0, 0, 0],
            "confidence": 0.95,
            "source": "annotation"
        })
    
    # Range
    if ship.get('range_n_miles'):
        kvs.append({
            "kv_id": f"{kv_id_base}_range",
            "field_name": "Range_n_miles",
            "raw_text": f"Range, n miles: {ship.get('range_n_miles')} {ship.get('range_speed', '')}".strip(),
            "value_normalized": ship.get('range_n_miles'),
            "units": "n miles",
            "qualifier": ship.get('range_speed'),
            "page_no": page_no,
            "bbox": [0, 0, 0, 0],
            "confidence": 0.95,
            "source": "annotation"
        })
    
    # Complement
    if ship.get('complement'):
        kvs.append({
            "kv_id": f"{kv_id_base}_complement",
            "field_name": "Complement",
            "raw_text": f"Complement: {ship.get('complement')}",
            "value_normalized": ship.get('complement'),
            "units": "persons",
            "page_no": page_no,
            "bbox": [0, 0, 0, 0],
            "confidence": 0.95,
            "source": "annotation"
        })
    
    return kvs

def process_pdf(pdf_path: str, mistral_client: Optional[Mistral] = None) -> str:
    """
    Main function to process PDF with Mistral OCR
    
    Args:
        pdf_path: Path to PDF file
        mistral_client: Mistral client instance (uses global client if None)
        
    Returns:
        Path to saved JSON file with usable format
    """
    # Process PDF with Mistral
    mistral_result = process_pdf_with_mistral(pdf_path, mistral_client)
    
    # Extract annotations in usable format
    annotations_data = extract_annotations(mistral_result)
    
    # Process images (keep only base64, no file saving)
    logger.info("Processing images from annotations (base64 only)...")
    annotations_data = extract_and_save_images(annotations_data, pdf_path)
    
    # Create output structure
    output_data = {
        "source_file": os.path.basename(pdf_path),
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "document_annotation": annotations_data["document_annotation"],
        "bbox_annotations": annotations_data["bbox_annotations"],
        "full_text": annotations_data["text"],
        "pages": annotations_data["pages"]
    }
    
    # Save results
    ensure_dir(OUTPUT_DIR)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved annotations to {output_path}")
    
    # Add full page images to JSON
    logger.info("Adding full page images to JSON...")
    add_page_images_to_json(output_path, pdf_path)
    
    # Generate viewer.html with correct JSON filename
    json_filename = os.path.basename(output_path)
    viewer_path = generate_viewer_html(json_filename, OUTPUT_DIR)
    
    # Also save raw result for debugging (optional, can be disabled)
    # raw_path = os.path.join(OUTPUT_DIR, f"{base_name}_raw.json")
    # try:
    #     if hasattr(mistral_result, '__dict__'):
    #         raw_dict = mistral_result.__dict__
    #     elif hasattr(mistral_result, 'model_dump'):
    #         raw_dict = mistral_result.model_dump()
    #     elif isinstance(mistral_result, dict):
    #         raw_dict = mistral_result
    #     else:
    #         raw_dict = {"result": str(mistral_result)}
    #     
    #     with open(raw_path, "w", encoding="utf-8") as f:
    #         json.dump(raw_dict, f, indent=2, ensure_ascii=False, default=str)
    #     
    #     logger.info(f"Saved raw result to {raw_path}")
    # except Exception as e:
    #     logger.warning(f"Could not save raw result: {e}")
    
    return output_path, viewer_path


# ---- CLI ----
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process PDF with Mistral Document AI OCR"
    )
    parser.add_argument(
        "pdf",
        help="Input PDF file path"
    )
    parser.add_argument(
        "--api-key",
        help="Mistral API key (overrides environment variable)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        parser.error(
            "MISTRAL_API_KEY not found. Please set it as an environment variable, "
            "create a .env file with MISTRAL_API_KEY=your_key, or use --api-key"
        )
    
    # Initialize Mistral client
    mistral_client = Mistral(api_key=api_key)
    
    try:
        result = process_pdf(args.pdf, mistral_client)
        json_path, viewer_path = result if isinstance(result, tuple) else (result, None)
        
        print(f"\n✓ OCR processing with annotations completed successfully!")
        print(f"Annotations saved to: {json_path}")
        if viewer_path:
            print(f"Viewer HTML generated: {viewer_path}")
            print(f"\nTo view the document:")
            print(f"  1. cd {OUTPUT_DIR}")
            print(f"  2. python -m http.server 8000")
            print(f"  3. Open http://localhost:8000/viewer.html in your browser")
        print(f"\nThe output contains:")
        print(f"  - document_annotation: Structured ship data extracted from the page")
        print(f"  - bbox_annotations: Annotated figures/images")
        print(f"  - full_text: Extracted text from the document")
        print(f"  - page_image_base64: Full page images embedded in JSON")
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        raise

