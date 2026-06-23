import pytest
import os
import json
import re
from unittest.mock import MagicMock, patch
import anthropic
from anthropic import RateLimitError

from models import Clause
from llm_client import LLMClient
from agent import ContractAgent, LLMResponse
import prompts
from rag_engine import RAGEngine
from risk_scorer import RiskScorer

# Mock classes for Anthropic response structure
class MockUsage:
    def __init__(self, input_tokens=100, output_tokens=150):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class MockTextBlock:
    def __init__(self, text):
        self.text = text
        self.type = "text"

class MockToolUseBlock:
    def __init__(self, id, name, input_args):
        self.id = id
        self.name = name
        self.input = input_args
        self.type = "tool_use"

class MockMessage:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.usage = MockUsage()
        self.stop_reason = stop_reason

# 1. Test Token Guard
def test_token_guard():
    client = LLMClient(api_key="dummy_key", token_limit=10)
    
    # Tiny prompt fits
    assert client.count_tokens("Hello") < 10
    
    # Prompt exceeding limit should raise ValueError
    with pytest.raises(ValueError, match="exceeds token guard limit"):
        client.create_message(messages=[{"role": "user", "content": "This is a prompt that is definitely longer than ten tokens"}])

# 2. Test Retry Mechanism on Transient Errors
@patch("anthropic.Anthropic")
def test_retry_mechanism(mock_anthropic):
    # Setup mock client that fails twice and succeeds on third attempt
    mock_instance = MagicMock()
    mock_anthropic.return_value = mock_instance
    
    # Create mock error
    # anthropic exceptions need a mock response and request object
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 429
    
    transient_error = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={}
    )
    
    success_response = MockMessage(content=[MockTextBlock("Success")])
    
    # Side effect: error, error, success
    mock_instance.messages.create.side_effect = [
        transient_error,
        transient_error,
        success_response
    ]
    
    client = LLMClient(api_key="dummy_key")
    res = client.create_message(messages=[{"role": "user", "content": "test"}])
    
    assert res.content[0].text == "Success"
    assert mock_instance.messages.create.call_count == 3

# 3. Test Disclaimer Validation
def test_disclaimer_validation():
    agent = ContractAgent()
    
    # Valid disclaimer matches prompts.DISCLAIMER_TEXT
    valid_json = json.dumps({
        "original_clause_summary": "Summary.",
        "redlined_clause": "The modified clause text.",
        "redline_rationale": "Because of risk.",
        "negotiation_priority": "MUST_CHANGE",
        "walk_away_trigger": "None.",
        "confidence_score": 0.9,
        "legal_disclaimer": prompts.DISCLAIMER_TEXT
    })
    
    res = agent._parse_and_validate(valid_json)
    assert res.legal_disclaimer == prompts.DISCLAIMER_TEXT
    
    # Invalid disclaimer should raise ValueError
    invalid_json = json.dumps({
        "original_clause_summary": "Summary.",
        "redlined_clause": "The modified clause text.",
        "redline_rationale": "Because of risk.",
        "negotiation_priority": "MUST_CHANGE",
        "walk_away_trigger": "None.",
        "confidence_score": 0.9,
        "legal_disclaimer": "Invalid disclaimer text."
    })
    
    with pytest.raises(ValueError, match="Disclaimer text mismatch"):
        agent._parse_and_validate(invalid_json)

# 4. Test RAG Fallback
def test_rag_fallback():
    # Force agent to fallback by passing a dummy client that raises exception
    dummy_client = LLMClient(api_key="dummy_key")
    dummy_client.create_message = MagicMock(side_effect=RuntimeError("API failure"))
    
    agent = ContractAgent(llm_client=dummy_client)
    
    clause = Clause(
        clause_id="TEST_001_CLS_01",
        raw_text="The buyer shall solely pay for everything unilaterally.",
        governing_law_jurisdiction="Delaware",
        contract_value_usd=100000.0,
        clause_type="Liability",
        risk_flag="HIGH_RISK"
    )
    
    # Reviewing clause should fall back to RAG suggestion and succeed
    res = agent.review_clause(clause)
    
    assert isinstance(res, LLMResponse)
    assert res.legal_disclaimer == prompts.DISCLAIMER_TEXT
    assert "Fallback" in res.redline_rationale
    assert len(res.redlined_clause) > 0

# 5. Test ReAct Tool Loop Execution
def test_react_tool_execution():
    agent = ContractAgent()
    
    # Mock LLMClient to call precedent_lookup tool, then return final text
    mock_client = MagicMock()
    agent.llm_client = mock_client
    
    # Turn 1: request tool call 'precedent_lookup'
    tool_use_block = MockToolUseBlock(
        id="toolu_123",
        name="precedent_lookup",
        input_args={"query": "governing law"}
    )
    message_1 = MockMessage(content=[tool_use_block])
    
    # Turn 2: return final text response containing JSON
    final_json = json.dumps({
        "original_clause_summary": "Summary.",
        "redlined_clause": "Propose New York law.",
        "redline_rationale": "Because we found NY precedents.",
        "negotiation_priority": "SHOULD_CHANGE",
        "walk_away_trigger": "None.",
        "confidence_score": 0.85,
        "legal_disclaimer": prompts.DISCLAIMER_TEXT
    })
    message_2 = MockMessage(content=[MockTextBlock(final_json)])
    
    mock_client.create_message.side_effect = [message_1, message_2]
    
    clause = Clause(
        clause_id="TEST_001_CLS_01",
        raw_text="This agreement is governed by the laws of Ohio.",
        governing_law_jurisdiction="Ohio",
        contract_value_usd=10000.0,
        clause_type="Governing_Law",
        risk_flag="HIGH_RISK"
    )
    
    res = agent.review_clause(clause)
    
    assert res.redlined_clause == "Propose New York law."
    assert res.legal_disclaimer == prompts.DISCLAIMER_TEXT
    assert mock_client.create_message.call_count == 2

# 6. Contract Pipeline Review and Edge Case validation on all 10 Contracts
def test_review_all_10_contracts():
    agent = ContractAgent()
    contracts_dir = "contracts"
    all_files = os.listdir(contracts_dir)
    contract_files = [f for f in all_files if f.endswith(".pdf") or f.endswith(".docx")]
    
    assert len(contract_files) == 10, f"Expected 10 contracts, found {len(contract_files)}"
    
    output_dir = "contracts/redlines"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in contract_files:
        filepath = os.path.join(contracts_dir, filename)
        contract_id, _ = os.path.splitext(filename)
        
        # Review contract
        redlines = agent.review_contract(filepath)
        
        # Save output redlines JSON
        output_path = os.path.join(output_dir, f"{contract_id}_redlines.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(redlines, f, indent=2)
            
        # Verify disclaimer compliance for all returned redlines
        for redline in redlines:
            assert redline["legal_disclaimer"] == prompts.DISCLAIMER_TEXT, (
                f"Disclaimer mismatch in contract {contract_id}, clause {redline['clause_id']}"
            )
            assert len(redline["redlined_clause"]) > 0
            assert len(redline["redline_rationale"]) > 0
            
        print(f"Contract {contract_id}: reviewed {len(redlines)} high-risk clauses.")
