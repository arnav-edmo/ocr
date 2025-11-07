# Mistral Document AI OCR

This script processes PDF files using Mistral Document AI OCR API.

## Setup

1. **Activate your virtual environment:**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   
   Option 1: Create a `.env` file (recommended):
   ```bash
   cp .env.example .env
   # Then edit .env and add your Mistral API key
   ```
   
   Option 2: Set environment variable:
   ```bash
   export MISTRAL_API_KEY=your_api_key_here
   ```
   
   Get your API key from: https://console.mistral.ai/

## Usage

Process a PDF file:

```bash
python mistral_ocr.py "Copy of Janes Fighting Ships 2023-2024.pdf"
```

Or with API key as argument:

```bash
python mistral_ocr.py "Copy of Janes Fighting Ships 2023-2024.pdf" --api-key your_api_key
```

## Output

The script will create:
- `output/{filename}_annotations.json` - **Usable format** with structured annotations:
  - `document_annotation`: Extracted ship data (classes, specifications, country info, force details)
  - `bbox_annotations`: Annotated figures/images with descriptions
  - `full_text`: Complete extracted text
  - `pages`: Page-level data if available
- `output/{filename}_raw.json` - Raw response from Mistral API (for debugging)

## How it works

1. Uploads the PDF to Mistral's cloud storage
2. Processes the PDF with Mistral OCR API with **Annotations**:
   - **Document Annotation**: Extracts structured ship information matching the page structure
   - **BBox Annotation**: Annotates figures/images found in the document
3. Extracts annotations into a clean, usable JSON format
4. Saves annotations and raw results

## Annotations

The script uses Mistral Document AI Annotations to extract structured data matching the Janes Fighting Ships page structure:

### Document Annotation (`PageAnnotation`)
Extracts complete page structure:
- **Country Overview**: Political info, geography, area, coastline, ports, capital
- **Force Details**: Force name, territorial waters, personnel, commander, bases
- **Ship Classes**: For each class:
  - Class name and type (e.g., "DAMEN STAN PATROL 4207 CLASS (PB)")
  - Individual ships (name, pennant number, ex-number, launch/commission dates)
  - Specifications (displacement, dimensions, speed, range, complement)
  - Machinery, guns, radars
  - Commentary section

### BBox Annotation (`FigureAnnotation`)
For each image/figure:
- Image type (photograph, diagram, chart)
- Caption and ship identification
- Detailed description
- Photo credit if available

The output is in a **usable JSON format** that directly matches the page structure - no complex canonical conversion needed!

## Notes

- The PDF is temporarily uploaded to Mistral's cloud storage during processing
- The uploaded file is automatically deleted after processing
- Large PDFs may take some time to process
- Make sure you have sufficient API credits/quota

