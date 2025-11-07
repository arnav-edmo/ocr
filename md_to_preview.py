#!/usr/bin/env python3
"""
Script to convert markdown files with literal \n characters to properly formatted markdown.
Handles \n escape sequences and converts them to actual newlines.
"""

import sys
import os
import argparse


def convert_newlines(content):
    """Convert literal \\n strings to actual newlines."""
    # Replace literal \n with actual newlines
    content = content.replace('\\n', '\n')
    return content


def process_markdown_file(input_file, output_file=None):
    """
    Process a markdown file, converting literal \n to newlines.
    
    Args:
        input_file: Path to input markdown file
        output_file: Path to output file (optional, defaults to input_file_preview.md)
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert literal \n to actual newlines
    converted_content = convert_newlines(content)
    
    # Determine output file name
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_preview.md"
    
    # Write the converted content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(converted_content)
    
    print(f"✓ Converted '{input_file}' to '{output_file}'")
    print(f"  Preview file created with {len(converted_content.splitlines())} lines")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert markdown files with literal \\n characters to properly formatted markdown'
    )
    parser.add_argument(
        'input_file',
        help='Input markdown file to process'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: input_file_preview.md)'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify the input file in place (overwrites original)'
    )
    
    args = parser.parse_args()
    
    if args.in_place:
        # Create a temporary file, then replace original
        import tempfile
        import shutil
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            converted = convert_newlines(content)
            tmp.write(converted)
            tmp_path = tmp.name
        
        shutil.move(tmp_path, args.input_file)
        print(f"✓ Converted '{args.input_file}' in place")
    else:
        process_markdown_file(args.input_file, args.output)


if __name__ == '__main__':
    main()

