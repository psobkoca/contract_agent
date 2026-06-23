import os
import re
import csv
import json
import sys
import argparse
from typing import List, Optional
from loguru import logger

# Import standard models and config
from models import Clause
from config import config

class ParserError(Exception):
    """Custom exception raised when contract parsing fails or file is empty/unreadable."""
    pass

# Section header regex patterns
SECTION_NUM_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+(.*)$')
WORD_SECTION_PATTERN = re.compile(r'^(Section|SECTION|Clause|CLAUSE)\s+([A-Za-z0-9\.]+)\.?:?\s*(.*)$')
ALL_CAPS_PATTERN = re.compile(r'^[A-Z0-9\s_,\-\(\)\&\:]{4,60}$')

def is_section_header(text: str) -> bool:
    """Checks if a given line of text matches a section header pattern."""
    text = text.strip()
    if not text:
        return False
        
    # Ignore standard signature block lines, footers, etc.
    if (re.search(r'Page \d+', text) or 
        text.endswith('- CONFIDENTIAL') or 
        'By: __' in text or 
        text.startswith('Name:') or 
        text.startswith('Title:') or 
        text.startswith('By:') or
        text.startswith('PARTY A:') or
        text.startswith('PARTY B:')):
        return False
        
    if SECTION_NUM_PATTERN.match(text):
        return True
    if WORD_SECTION_PATTERN.match(text):
        return True
    if ALL_CAPS_PATTERN.match(text) and len(text) > 5 and text not in ["PARTY A:", "PARTY B:"]:
        return True
        
    return False

def load_contract_metadata(contract_id: str) -> Optional[dict]:
    """Loads contract metadata from contract_metadata.csv for the given contract_id."""
    metadata_path = os.path.join("contracts", "contract_metadata.csv")
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found at {metadata_path}")
        return None
        
    try:
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("contract_id") == contract_id:
                    return row
    except Exception as e:
        logger.error(f"Error reading contract metadata: {e}")
        
    return None

def extract_lines_from_pdf(filepath: str) -> List[str]:
    """Extracts lines of text from a PDF file using pdfplumber."""
    import pdfplumber
    lines = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines.extend(text.split("\n"))
    except Exception as e:
        raise ParserError(f"PDF file is unreadable: {filepath}. Underlying error: {e}")
    return lines

def extract_lines_from_docx(filepath: str) -> List[str]:
    """Extracts paragraphs of text from a DOCX file using python-docx."""
    from docx import Document
    try:
        doc = Document(filepath)
        return [p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        raise ParserError(f"DOCX file is unreadable: {filepath}. Underlying error: {e}")

def segment_text_two_pass(blocks: List[str], contract_id: str, is_docx: bool) -> List[Clause]:
    """Segments contract text blocks into Clause objects using a two-pass approach."""
    # Pass 1: Scan and identify all section headers and their indices
    header_indices = []
    for i, block in enumerate(blocks):
        if is_section_header(block):
            header_indices.append(i)
            
    clauses = []
    clause_index = 1
    
    # Pass 2: Extract content segments between headers and split by paragraph boundaries
    # We define segment boundaries: [(section_title, start_index, end_index), ...]
    segments = []
    if not header_indices:
        # No headers found; treat the whole document as Introduction
        segments.append(("Introduction", 0, len(blocks)))
    else:
        # Text before the first header goes to "Introduction"
        if header_indices[0] > 0:
            segments.append(("Introduction", 0, header_indices[0]))
            
        for idx, start in enumerate(header_indices):
            title = blocks[start].strip()
            end = header_indices[idx + 1] if idx + 1 < len(header_indices) else len(blocks)
            # Content of this segment starts after the header block itself
            segments.append((title, start + 1, end))
            
    # Process each segment
    for title, start, end in segments:
        segment_blocks = blocks[start:end]
        
        # Split segment into paragraphs
        paragraphs = []
        if is_docx:
            # DOCX blocks are already paragraphs
            paragraphs = [b.strip() for b in segment_blocks if b.strip()]
        else:
            # PDF lines need joining into paragraphs
            current_para = []
            for b in segment_blocks:
                b = b.strip()
                if not b:
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
                else:
                    current_para.append(b)
            if current_para:
                paragraphs.append(" ".join(current_para))
                
        # Create Clause objects for paragraphs matching minimum length
        for para in paragraphs:
            # Skip signature blocks, page counts, or lines that got grouped incorrectly
            if (para.endswith('- CONFIDENTIAL') or 
                'By: __' in para or 
                para.startswith('Name:') or 
                para.startswith('Title:') or
                para.startswith('PARTY A:') or 
                para.startswith('PARTY B:')):
                continue
                
            if len(para) >= config.parsing.min_clause_chars:
                clauses.append(Clause(
                    clause_id=f"{contract_id}_CLS_{clause_index:03d}",
                    section_number=title,
                    raw_text=para
                ))
                clause_index += 1
                
    return clauses

def parse_contract(filepath: str) -> List[Clause]:
    """Detects format, extracts text, enriches with metadata, and segments into Clause objects."""
    if not os.path.exists(filepath):
        raise ParserError(f"Contract file not found at: {filepath}")
        
    filename = os.path.basename(filepath)
    contract_id, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # 1. Format Detection & Extraction
    logger.info(f"Extracting text from {filename}...")
    is_docx = False
    if ext == ".pdf":
        blocks = extract_lines_from_pdf(filepath)
    elif ext == ".docx":
        is_docx = True
        blocks = extract_lines_from_docx(filepath)
    else:
        raise ParserError(f"Unsupported file format: {ext}")
        
    # Abort if the file is empty after extraction
    clean_blocks = [b.strip() for b in blocks if b.strip()]
    if not clean_blocks:
        raise ParserError(f"Contract file is empty or unreadable after text extraction: {filepath}")
        
    # 2. Two-Pass Segmentation
    logger.info("Segmenting text into clauses using two-pass parser...")
    clauses = segment_text_two_pass(blocks, contract_id, is_docx)
    
    # 3. Enrichment with metadata
    logger.info(f"Enriching clauses with metadata registry for contract: {contract_id}")
    metadata = load_contract_metadata(contract_id)
    
    if metadata:
        logger.success(f"Matched Metadata: Type={metadata.get('contract_type')}, Counterparty={metadata.get('counterparty_name')}")
        # Map values to each Clause
        for clause in clauses:
            clause.counterparty_name = metadata.get("counterparty_name")
            clause.contract_type = metadata.get("contract_type")
            clause.governing_law_jurisdiction = metadata.get("governing_law")
            clause.effective_date = metadata.get("effective_date")
            try:
                clause.contract_value_usd = float(metadata.get("contract_value_usd", 0.0))
            except ValueError:
                clause.contract_value_usd = 0.0
    else:
        logger.warning(f"Log Warning: contract_id '{contract_id}' not found in metadata registry. Continuing with null metadata.")
        
    return clauses

def main():
    parser = argparse.ArgumentParser(description="Contract Clause Parser and Metadata Enricher")
    parser.add_argument("--contract", required=True, help="Path to the PDF/DOCX contract file.")
    parser.add_argument("--output", help="Optional path to output the results as a JSON file.")
    
    args = parser.parse_args()
    
    try:
        clauses = parse_contract(args.contract)
        
        if not clauses:
            logger.warning("No clauses extracted from the contract.")
            sys.exit(0)
            
        # Calculate contract-level statistics (FR-04)
        total_clause_count = len(clauses)
        word_count = sum(len(c.raw_text.split()) for c in clauses)
        unique_section_count = len(set(c.section_number for c in clauses))
        contract_type = clauses[0].contract_type or "Unknown"
        
        # Log stats using loguru
        logger.success("--- Contract-Level Statistics ---")
        logger.info(f"Contract Type: {contract_type}")
        logger.info(f"Total Clause Count: {total_clause_count}")
        logger.info(f"Word Count: {word_count}")
        logger.info(f"Unique Section Count: {unique_section_count}")
        
        contract_summary = {
            "total_clause_count": total_clause_count,
            "word_count": word_count,
            "unique_section_count": unique_section_count,
            "contract_type": contract_type
        }
        
        # Write to JSON output if requested
        if args.output:
            output_data = {
                "contract_summary": contract_summary,
                "clauses": [c.model_dump() for c in clauses]
            }
            try:
                with open(args.output, mode="w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)
                logger.success(f"Successfully saved run output to {args.output}")
            except Exception as e:
                logger.error(f"Failed to write output file: {e}")
                
    except ParserError as pe:
        logger.critical(f"Parser Aborted: {pe}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
