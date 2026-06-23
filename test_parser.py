import pytest
import os
from contract_parser import parse_contract, ParserError
from models import Clause

def test_parse_valid_pdf():
    filepath = "contracts/CTR_001.pdf"
    assert os.path.exists(filepath), "CTR_001.pdf does not exist"
    
    clauses = parse_contract(filepath)
    assert len(clauses) > 0, "No clauses extracted from PDF"
    
    # Check clause fields
    for c in clauses:
        assert isinstance(c, Clause)
        assert c.clause_id.startswith("CTR_001_CLS_")
        assert c.raw_text is not None
        assert len(c.raw_text) > 0
        assert c.contract_type == "NDA"
        assert c.counterparty_name == "Zephyr Robotics Inc."
        assert c.governing_law_jurisdiction == "Delaware"

def test_parse_valid_docx():
    filepath = "contracts/CTR_002.docx"
    assert os.path.exists(filepath), "CTR_002.docx does not exist"
    
    clauses = parse_contract(filepath)
    assert len(clauses) > 0, "No clauses extracted from DOCX"
    
    # Check clause fields
    for c in clauses:
        assert isinstance(c, Clause)
        assert c.clause_id.startswith("CTR_002_CLS_")
        assert c.raw_text is not None
        assert len(c.raw_text) > 0
        assert c.contract_type == "NDA"
        assert c.counterparty_name == "Aurora Pharmaceuticals Inc."
        assert c.governing_law_jurisdiction == "Massachusetts"

def test_parse_nonexistent_file():
    filepath = "contracts/nonexistent_contract_file.pdf"
    with pytest.raises(ParserError):
        parse_contract(filepath)

def test_parse_unsupported_format():
    # Create a dummy unsupported file
    filepath = "contracts/dummy_file.txt"
    with open(filepath, "w") as f:
        f.write("Some text here")
        
    try:
        with pytest.raises(ParserError):
            parse_contract(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
