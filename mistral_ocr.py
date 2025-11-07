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
    Extract base64 images from annotations and save them to files.
    
    Args:
        annotations_data: Dictionary containing bbox_annotations with image_base64 fields
        pdf_path: Original PDF path for naming output directory
        
    Returns:
        Updated annotations_data with image_file_path added to each annotation
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images_dir = os.path.join(OUTPUT_DIR, f"{base_name}_images")
    ensure_dir(images_dir)
    
    saved_images = []
    image_counter = 0
    
    # Extract images from bbox_annotations
    bbox_annotations = annotations_data.get("bbox_annotations", [])
    
    for idx, bbox_ann in enumerate(bbox_annotations):
        image_base64 = None
        
        # Try to get image_base64 from various possible locations
        if isinstance(bbox_ann, dict):
            image_base64 = bbox_ann.get('image_base64')
        elif hasattr(bbox_ann, 'image_base64'):
            image_base64 = bbox_ann.image_base64
        
        if image_base64:
            try:
                # Handle data URI format: "data:image/jpeg;base64,..."
                if isinstance(image_base64, str):
                    # Check if it's a data URI
                    if image_base64.startswith('data:'):
                        # Extract the base64 part after the comma
                        match = re.match(r'data:image/(\w+);base64,(.+)', image_base64)
                        if match:
                            image_format = match.group(1)  # jpeg, png, etc.
                            base64_data = match.group(2)
                        else:
                            # Fallback: try to extract after first comma
                            parts = image_base64.split(',', 1)
                            if len(parts) == 2:
                                base64_data = parts[1]
                                # Try to detect format from mime type
                                if 'jpeg' in parts[0] or 'jpg' in parts[0]:
                                    image_format = 'jpeg'
                                elif 'png' in parts[0]:
                                    image_format = 'png'
                                else:
                                    image_format = 'jpeg'  # default
                            else:
                                base64_data = image_base64
                                image_format = 'jpeg'  # default
                    else:
                        # Assume it's already base64 without data URI prefix
                        base64_data = image_base64
                        image_format = 'jpeg'  # default
                    
                    # Decode base64
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Determine file extension
                    ext = image_format if image_format in ['jpeg', 'png', 'gif', 'webp'] else 'jpg'
                    if ext == 'jpeg':
                        ext = 'jpg'
                    
                    # Generate filename
                    image_counter += 1
                    filename = f"img_{image_counter:03d}.{ext}"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    
                    # Update annotation with file path
                    if isinstance(bbox_ann, dict):
                        bbox_ann['image_file_path'] = filepath
                        bbox_ann['image_filename'] = filename
                    else:
                        # If it's an object, try to set attribute
                        try:
                            bbox_ann.image_file_path = filepath
                            bbox_ann.image_filename = filename
                        except:
                            pass
                    
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
    pages = annotations_data.get("pages", [])
    for page_idx, page in enumerate(pages):
        image_anns = []
        if isinstance(page, dict):
            # Check both 'images' and 'image_annotations' fields
            image_anns = page.get('images', [])
            if not image_anns:
                image_anns = page.get('image_annotations', [])
        elif hasattr(page, 'images'):
            image_anns = page.images
        elif hasattr(page, 'image_annotations'):
            image_anns = page.image_annotations
        else:
            continue
        
        for img_ann in image_anns:
            image_base64 = None
            if isinstance(img_ann, dict):
                image_base64 = img_ann.get('image_base64')
            elif hasattr(img_ann, 'image_base64'):
                image_base64 = img_ann.image_base64
            
            if image_base64:
                try:
                    # Same extraction logic as above
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
                    filepath = os.path.join(images_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    
                    if isinstance(img_ann, dict):
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
    
    logger.info(f"Extracted and saved {len(saved_images)} images to {images_dir}")
    annotations_data['extracted_images'] = saved_images
    annotations_data['images_directory'] = images_dir
    
    return annotations_data


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


def convert_mistral_to_canonical(mistral_result: Any, pdf_path: str) -> Dict[str, Any]:
    """
    Convert Mistral OCR result to canonical JSON format
    
    Args:
        mistral_result: Result from Mistral OCR API (can be dict or object)
        pdf_path: Original PDF file path
        
    Returns:
        Canonical JSON structure matching format.json
    """
    logger.info("Converting Mistral result to canonical format...")
    
    # Extract document-level information
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    source_file = os.path.basename(pdf_path)
    
    # Extract annotations
    doc_annotation, bbox_annotations = extract_annotations_from_result(mistral_result)
    logger.info(f"Extracted document annotation: {doc_annotation is not None}, bbox annotations: {len(bbox_annotations)}")
    
    # Convert result to dict if it's an object
    if not isinstance(mistral_result, dict):
        if hasattr(mistral_result, '__dict__'):
            result_dict = mistral_result.__dict__
        elif hasattr(mistral_result, 'model_dump'):
            result_dict = mistral_result.model_dump()
        else:
            # Try to access common attributes
            result_dict = {
                'text': getattr(mistral_result, 'text', ''),
                'pages': getattr(mistral_result, 'pages', []),
                'blocks': getattr(mistral_result, 'blocks', []),
            }
    else:
        result_dict = mistral_result
    
    # Initialize canonical structure
    canonical = {
        "document_id": doc_id,
        "source_file": source_file,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pages": [],
        "global_entities": [],
        "attachments": {
            "images": [],
            "raw_pdf": source_file
        },
        "indexing": {
            "full_text": result_dict.get('text', '') if isinstance(result_dict, dict) else '',
            "facets": {}
        },
        "annotations": {
            "document_annotation": doc_annotation,
            "bbox_annotations": bbox_annotations
        },
        "audit": {
            "pipeline_version": "mistral_document_ai_v1.0",
            "created_by": "mistral_ocr.py",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "human_reviewed": False,
            "human_reviews": []
        }
    }
    
    # Extract full text if available
    full_text = result_dict.get('text', '') if isinstance(result_dict, dict) else ''
    if not full_text and hasattr(mistral_result, 'text'):
        full_text = mistral_result.text
    
    # Process pages from Mistral result
    pages = result_dict.get('pages', []) if isinstance(result_dict, dict) else []
    if not pages and hasattr(mistral_result, 'pages'):
        pages = mistral_result.pages
    
    # If no pages structure, create a single page from the text
    if not pages and full_text:
        logger.info("No page structure found, creating single page from extracted text")
        page_entry = {
            "page_no": 1,
            "width": 0,
            "height": 0,
            "image": "images/page_0001.png",
            "layout_blocks": [
                {
                    "block_id": "p1_b1",
                    "type": "paragraph",
                    "bbox": [0, 0, 0, 0],
                    "text": full_text,
                    "lines": [],
                    "confidence": 0.95
                }
            ],
            "tables": [],
            "figures": [],
            "key_values": [],
            "entities": [],
            "full_text": full_text,
            "page_confidence": 0.95
        }
        
        # Add annotation data to page
        if doc_annotation:
            if isinstance(doc_annotation, dict):
                ships = doc_annotation.get('ships', [])
            else:
                ships = getattr(doc_annotation, 'ships', [])
            
            # Convert ship specs to key_values
            for ship in ships:
                if isinstance(ship, dict):
                    ship_dict = ship
                else:
                    ship_dict = ship.model_dump() if hasattr(ship, 'model_dump') else ship.__dict__
                
                page_entry["key_values"].extend(convert_ship_specs_to_key_values(ship_dict, 1))
                
                # Add ship names as entities
                ship_names = ship_dict.get('ship_names', [])
                for name in ship_names:
                    page_entry["entities"].append({
                        "entity_id": f"p1_e_{uuid.uuid4().hex[:4]}",
                        "type": "ship_name",
                        "text": name,
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.95
                    })
        
        # Update facets from document annotation
        if doc_annotation:
            if isinstance(doc_annotation, dict):
                country = doc_annotation.get('country')
                section_type = doc_annotation.get('section_type')
            else:
                country = getattr(doc_annotation, 'country', None)
                section_type = getattr(doc_annotation, 'section_type', None)
            
            if country:
                canonical["indexing"]["facets"]["country"] = country
            if section_type:
                canonical["indexing"]["facets"]["document_type"] = section_type
        
        canonical["pages"].append(page_entry)
    else:
        # Process structured pages
        for page_idx, page in enumerate(pages, start=1):
            # Handle both dict and object
            if not isinstance(page, dict):
                page = {
                    'width': getattr(page, 'width', 0),
                    'height': getattr(page, 'height', 0),
                    'blocks': getattr(page, 'blocks', []),
                    'text': getattr(page, 'text', ''),
                }
            
            page_entry = {
                "page_no": page_idx,
                "width": page.get('width', 0),
                "height": page.get('height', 0),
                "image": f"images/page_{page_idx:04d}.png",
                "layout_blocks": [],
                "tables": [],
                "figures": [],
                "key_values": [],
                "entities": [],
                "page_confidence": 0.95
            }
            
            # Extract text blocks
            blocks = page.get('blocks', [])
            page_text_parts = []
            
            for block_idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    block = {
                        'text': getattr(block, 'text', ''),
                        'bbox': getattr(block, 'bbox', [0, 0, 0, 0]),
                        'type': getattr(block, 'type', 'paragraph'),
                        'confidence': getattr(block, 'confidence', 0.95),
                    }
                
                block_text = block.get('text', '')
                bbox = block.get('bbox', [0, 0, 0, 0])
                
                if block_text:
                    page_text_parts.append(block_text)
                
                block_entry = {
                    "block_id": f"p{page_idx}_b{block_idx}",
                    "type": block.get('type', 'paragraph'),
                    "bbox": bbox,
                    "text": block_text,
                    "lines": [],
                    "confidence": block.get('confidence', 0.95)
                }
                
                # Extract lines if available
                lines = block.get('lines', [])
                for line_idx, line in enumerate(lines):
                    if not isinstance(line, dict):
                        line = {
                            'text': getattr(line, 'text', ''),
                            'bbox': getattr(line, 'bbox', [0, 0, 0, 0]),
                            'confidence': getattr(line, 'confidence', 0.95),
                            'words': getattr(line, 'words', []),
                        }
                    
                    line_entry = {
                        "line_id": f"p{page_idx}_b{block_idx}_l{line_idx}",
                        "text": line.get('text', ''),
                        "bbox": line.get('bbox', [0, 0, 0, 0]),
                        "words": [],
                        "ocr_confidence": line.get('confidence', 0.95)
                    }
                    
                    # Extract words if available
                    words = line.get('words', [])
                    for word in words:
                        if not isinstance(word, dict):
                            word = {
                                'text': getattr(word, 'text', ''),
                                'bbox': getattr(word, 'bbox', [0, 0, 0, 0]),
                                'confidence': getattr(word, 'confidence', 0.95),
                            }
                        
                        word_entry = {
                            "text": word.get('text', ''),
                            "bbox": word.get('bbox', [0, 0, 0, 0]),
                            "confidence": word.get('confidence', 0.95)
                        }
                        line_entry["words"].append(word_entry)
                    
                    block_entry["lines"].append(line_entry)
                
                page_entry["layout_blocks"].append(block_entry)
            
            # Extract tables if available
            tables = page.get('tables', [])
            for table_idx, table in enumerate(tables):
                if not isinstance(table, dict):
                    table = {
                        'bbox': getattr(table, 'bbox', [0, 0, 0, 0]),
                        'cells': getattr(table, 'cells', []),
                        'n_rows': getattr(table, 'n_rows', 0),
                        'n_cols': getattr(table, 'n_cols', 0),
                    }
                
                table_entry = {
                    "table_id": f"p{page_idx}_t{table_idx}",
                    "bbox": table.get('bbox', [0, 0, 0, 0]),
                    "n_rows": table.get('n_rows', 0),
                    "n_cols": table.get('n_cols', 0),
                    "cells": [],
                    "extracted_kv": {}
                }
                
                cells = table.get('cells', [])
                for cell in cells:
                    if not isinstance(cell, dict):
                        cell = {
                            'r': getattr(cell, 'r', 0),
                            'c': getattr(cell, 'c', 0),
                            'text': getattr(cell, 'text', ''),
                            'bbox': getattr(cell, 'bbox', [0, 0, 0, 0]),
                            'confidence': getattr(cell, 'confidence', 0.95),
                        }
                    
                    cell_entry = {
                        "r": cell.get('r', 0),
                        "c": cell.get('c', 0),
                        "text": cell.get('text', ''),
                        "bbox": cell.get('bbox', [0, 0, 0, 0]),
                        "confidence": cell.get('confidence', 0.95)
                    }
                    table_entry["cells"].append(cell_entry)
                
                page_entry["tables"].append(table_entry)
            
            page_entry["full_text"] = "\n".join(page_text_parts) if page_text_parts else page.get('text', '')
            canonical["pages"].append(page_entry)
    
    # Update full text from pages if not already set
    if not canonical["indexing"]["full_text"]:
        canonical["indexing"]["full_text"] = " ".join([p.get("full_text", "") for p in canonical["pages"]])
    
    return canonical


def save_results(canonical_json: Dict[str, Any], pdf_path: str, raw_result: Optional[Dict[str, Any]] = None):
    """
    Save OCR results to files
    
    Args:
        canonical_json: Canonical format JSON
        pdf_path: Original PDF path
        raw_result: Raw Mistral API response (optional)
    """
    ensure_dir(OUTPUT_DIR)
    
    # Save canonical JSON
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    canonical_path = os.path.join(OUTPUT_DIR, f"{base_name}_mistral_canonical.json")
    
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(canonical_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved canonical JSON to {canonical_path}")
    
    # Save raw result if provided
    if raw_result:
        raw_path = os.path.join(OUTPUT_DIR, f"{base_name}_mistral_raw.json")
        try:
            # Try to serialize raw result
            if hasattr(raw_result, '__dict__'):
                raw_dict = raw_result.__dict__
            elif isinstance(raw_result, dict):
                raw_dict = raw_result
            else:
                raw_dict = {"result": str(raw_result)}
            
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved raw result to {raw_path}")
        except Exception as e:
            logger.warning(f"Could not save raw result: {e}")
    
    return canonical_path


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
    
    # Extract and save images from base64 data
    logger.info("Extracting and saving images from annotations...")
    annotations_data = extract_and_save_images(annotations_data, pdf_path)
    
    # Create output structure
    output_data = {
        "source_file": os.path.basename(pdf_path),
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "document_annotation": annotations_data["document_annotation"],
        "bbox_annotations": annotations_data["bbox_annotations"],
        "full_text": annotations_data["text"],
        "pages": annotations_data["pages"],
        "extracted_images": annotations_data.get("extracted_images", []),
        "images_directory": annotations_data.get("images_directory", "")
    }
    
    # Save results
    ensure_dir(OUTPUT_DIR)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved annotations to {output_path}")
    
    # Also save raw result for debugging
    raw_path = os.path.join(OUTPUT_DIR, f"{base_name}_raw.json")
    try:
        if hasattr(mistral_result, '__dict__'):
            raw_dict = mistral_result.__dict__
        elif hasattr(mistral_result, 'model_dump'):
            raw_dict = mistral_result.model_dump()
        elif isinstance(mistral_result, dict):
            raw_dict = mistral_result
        else:
            raw_dict = {"result": str(mistral_result)}
        
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved raw result to {raw_path}")
    except Exception as e:
        logger.warning(f"Could not save raw result: {e}")
    
    return output_path


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
        print(f"\n✓ OCR processing with annotations completed successfully!")
        print(f"Annotations saved to: {result}")
        print(f"\nThe output contains:")
        print(f"  - document_annotation: Structured ship data extracted from the page")
        print(f"  - bbox_annotations: Annotated figures/images")
        print(f"  - full_text: Extracted text from the document")
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        raise

