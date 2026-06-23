import os
import re
import csv
from typing import List, Optional
from loguru import logger
from tabulate import tabulate

# Import standard models
from models import Clause

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
        logger.error(f"Error reading PDF file {filepath}: {e}")
        raise e
    return lines

def extract_lines_from_docx(filepath: str) -> List[str]:
    """Extracts paragraphs of text from a DOCX file using python-docx."""
    from docx import Document
    try:
        doc = Document(filepath)
        return [p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        logger.error(f"Error reading DOCX file {filepath}: {e}")
        raise e

def segment_text_into_clauses(lines: List[str], contract_id: str) -> List[Clause]:
    """Segments contract lines/paragraphs into Clause objects using section-header rules."""
    clauses = []
    current_section = "Introduction"
    buffer = []
    clause_index = 1
    
    # Section header regex patterns
    section_num_pattern = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+(.*)$')
    word_section_pattern = re.compile(r'^(Section|SECTION|Clause|CLAUSE)\s+([A-Za-z0-9\.]+)\.?:?\s*(.*)$')
    all_caps_pattern = re.compile(r'^[A-Z0-9\s_,\-\(\)\&\:]{4,60}$')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Ignore page numbers, headers/footers, and standard signature block lines
        if (re.search(r'Page \d+', line) or 
            line.endswith('- CONFIDENTIAL') or 
            'By: __' in line or 
            line.startswith('Name:') or 
            line.startswith('Title:') or 
            line.startswith('By:') or
            line.startswith('PARTY A:') or
            line.startswith('PARTY B:')):
            continue
            
        is_header = False
        
        m_num = section_num_pattern.match(line)
        m_word = word_section_pattern.match(line)
        
        if m_num:
            is_header = True
        elif m_word:
            is_header = True
        elif all_caps_pattern.match(line) and len(line) > 5 and line not in ["PARTY A:", "PARTY B:"]:
            is_header = True
            
        if is_header:
            # If buffer contains accumulated body text, save it as a clause before starting a new section
            if buffer:
                text = " ".join(buffer).strip()
                if text:
                    clauses.append(Clause(
                        clause_id=f"{contract_id}_CLS_{clause_index:03d}",
                        section_number=current_section,
                        text=text
                    ))
                    clause_index += 1
                buffer = []
            current_section = line
        else:
            buffer.append(line)
            
    # Save the remaining clause in buffer
    if buffer:
        text = " ".join(buffer).strip()
        if text:
            clauses.append(Clause(
                clause_id=f"{contract_id}_CLS_{clause_index:03d}",
                section_number=current_section,
                text=text
            ))
            
    return clauses

def parse_contract(filepath: str) -> List[Clause]:
    """Detects format, extracts text, enriches with metadata, and segments into Clause objects."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return []
        
    filename = os.path.basename(filepath)
    contract_id, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # 1. Enrich with metadata lookup
    logger.info(f"Enriching metadata lookup for contract: {contract_id}")
    metadata = load_contract_metadata(contract_id)
    if metadata:
        logger.success(f"Matched Metadata: Type={metadata['contract_type']}, Counterparty={metadata['counterparty_name']}, Law={metadata['governing_law']}, Value=${metadata['contract_value_usd']}, Priority={metadata['review_priority']}")
    else:
        logger.warning(f"No metadata found in contract_metadata.csv for ID: {contract_id}")
        
    # 2. Extract lines based on format detection
    logger.info(f"Extracting text from {filename}...")
    if ext == ".pdf":
        lines = extract_lines_from_pdf(filepath)
    elif ext == ".docx":
        lines = extract_lines_from_docx(filepath)
    else:
        logger.error(f"Unsupported file format: {ext}")
        return []
        
    # 3. Segment into Clause objects
    logger.info("Segmenting text into clauses...")
    clauses = segment_text_into_clauses(lines, contract_id)
    logger.success(f"Successfully segmented {len(clauses)} clauses from {filename}")
    
    return clauses

def main():
    import sys
    # If a file is passed in CLI arguments, parse it
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        clauses = parse_contract(filepath)
        
        # Display the parsed clauses in a table
        table_data = []
        for c in clauses:
            # Truncate text for cleaner display in table
            truncated_text = c.raw_text[:80] + "..." if len(c.raw_text) > 80 else c.raw_text
            table_data.append([c.clause_id, c.section_number, truncated_text])
            
        print("\n--- Extracted Clauses Table ---")
        print(tabulate(table_data, headers=["Clause ID", "Section Header / Number", "Clause Text (Truncated)"], tablefmt="grid"))
    else:
        print("Usage: python contract_parser.py <path_to_contract_file>")
        print("Example: python contract_parser.py contracts/CTR_001.pdf")

if __name__ == "__main__":
    main()
