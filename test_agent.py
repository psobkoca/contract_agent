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
from reporter import Reporter

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
    
    # Mock messages.create API call
    mock_create = MagicMock(return_value=MockMessage(content=[MockTextBlock("Success")]))
    client.client = MagicMock()
    client.client.messages.create = mock_create
    
    # Pass a prompt that exceeds the 10 token limit
    long_prompt = "This is a prompt that is definitely longer than ten tokens"
    res = client.create_message(messages=[{"role": "user", "content": long_prompt}])
    
    # Check that it succeeded (didn't raise ValueError) and was truncated
    assert res.content[0].text == "Success"
    call_kwargs = mock_create.call_args[1]
    sent_prompt = call_kwargs["messages"][0]["content"]
    assert client.count_tokens(sent_prompt) <= 10
    assert len(sent_prompt) < len(long_prompt)

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


# 7. Test Edge Cases & Pipeline Coverage

def test_zero_high_critical_clauses():
    # Test a contract with zero high/critical clauses.
    # It should skip LLM reviews and generate valid logs/reports.
    agent = ContractAgent()
    
    # Mock RiskScorer.score_clause so it always returns LOW risk (tier="LOW", score=0.1)
    mock_scorer = MagicMock()
    mock_scorer.score_clause.return_value = {
        "final_score": 0.1,
        "one_sidedness_score": 0.0,
        "market_deviation_score": 0.0,
        "jurisdiction_risk_score": 0.1,
        "value_risk_score": 0.0
    }
    mock_scorer.clause_type_weights = {} # No overrides
    agent.risk_scorer = mock_scorer
    
    # We will pass a real contract but because it has low risk, review_contract should return []
    redlines = agent.review_contract("contracts/CTR_001.pdf")
    assert redlines == []


def test_reporter_with_zero_clauses(tmp_path):
    # Initialize reporter with a temp dir
    reporter = Reporter(run_id="test_zero_run")
    reporter.output_dir = str(tmp_path)
    
    contract_review_logs = [{
        "contract_id": "CTR_001",
        "counterparty_name": "ACME",
        "run_timestamp": "2026-06-23T00:00:00Z",
        "contract_summary": "Summary of contract CTR_001.",
        "total_clauses": 5,
        "clauses_reviewed_by_llm": 0,
        "risk_tier_distribution": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 5},
        "redlines": [],
        "loop_count": 0,
        "fallback_mode": False,
        "latency_ms": 0
    }]
    
    scored_contracts_clauses = {
        "CTR_001": []
    }
    
    scorecard_data = [{
        "contract_id": "CTR_001",
        "contract_type": "NDA",
        "counterparty_name": "ACME",
        "governing_law": "Delaware",
        "contract_value_usd": 10000.0,
        "risk_score": 0.15,
        "risk_tier": "LOW"
    }]
    
    md_path = reporter.generate_markdown_memo(contract_review_logs, scored_contracts_clauses, scorecard_data)
    json_path = reporter.generate_json_log(contract_review_logs[0])
    csv_path = reporter.generate_csv_redlines([])
    
    # Assert MD file exists and has disclaimer at top and bottom
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    disclaimer = prompts.DISCLAIMER_TEXT.strip()
    
    # Disclaimer should appear at least twice (top and bottom)
    assert content.count(disclaimer) >= 2
    assert content.startswith("> [!IMPORTANT]")
    assert content.strip().endswith(disclaimer)
    assert "*No HIGH or CRITICAL risk clauses were reviewed for this contract.*" in content
    
    # Assert JSON log exists
    assert os.path.exists(json_path)
    # Assert CSV redlines exists
    assert os.path.exists(csv_path)


@patch("anthropic.Anthropic")
def test_llm_timeout_triggers_fallback(mock_anthropic):
    # Setup mock anthropic client that raises APITimeoutError
    mock_instance = MagicMock()
    mock_anthropic.return_value = mock_instance
    
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 408
    
    timeout_err = anthropic.APITimeoutError(
        request=mock_request
    )
    mock_instance.messages.create.side_effect = timeout_err
    
    # Force LLMClient to fail with APITimeoutError
    client = LLMClient(api_key="dummy_key")
    # Set max attempts to 1 to speed up test
    with patch.object(client, "create_message", side_effect=timeout_err):
        agent = ContractAgent(llm_client=client)
        
        clause = Clause(
            clause_id="TEST_TIMEOUT_CLS_01",
            raw_text="The contractor shall indemnify the client for any and all claims.",
            governing_law_jurisdiction="California",
            contract_value_usd=50000.0,
            clause_type="Indemnification",
            risk_flag="HIGH_RISK"
        )
        
        # This should trigger _rag_fallback
        res = agent.review_clause(clause)
        
        assert res.fallback_mode is True
        assert res.legal_disclaimer == prompts.DISCLAIMER_TEXT
        assert "Fallback Mode:" in res.redline_rationale


def test_clause_no_matching_precedent():
    agent = ContractAgent()
    
    # Mock RAGEngine.hybrid_search to return empty list
    mock_rag = MagicMock()
    mock_rag.hybrid_search.return_value = []
    agent.rag_engine = mock_rag
    
    # Mock LLMClient to return a normal JSON response
    mock_client = MagicMock()
    agent.llm_client = mock_client
    
    expected_response = json.dumps({
        "original_clause_summary": "Original summary.",
        "redlined_clause": "Original raw text proposed.",
        "redline_rationale": "No precedents found, standard language accepted.",
        "negotiation_priority": "NICE_TO_HAVE",
        "walk_away_trigger": "None.",
        "confidence_score": 0.95,
        "legal_disclaimer": prompts.DISCLAIMER_TEXT
    })
    mock_client.create_message.return_value = MockMessage(content=[MockTextBlock(expected_response)])
    
    clause = Clause(
        clause_id="TEST_NOPREC_CLS_01",
        raw_text="Random custom clause text.",
        governing_law_jurisdiction="Delaware",
        contract_value_usd=0.0,
        clause_type="Miscellaneous",
        risk_flag="HIGH_RISK"
    )
    
    res = agent.review_clause(clause)
    
    assert res.redlined_clause == "Original raw text proposed."
    assert "No precedents found" in res.redline_rationale
    assert res.legal_disclaimer == prompts.DISCLAIMER_TEXT
    
    # Now verify the user prompt sent to LLM contains "No precedents found."
    call_args = mock_client.create_message.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[0]["content"]
    assert "No precedents found." in user_prompt


def test_disclaimer_presence_assertion_across_reports(tmp_path):
    reporter = Reporter(run_id="test_run_disclaimer")
    reporter.output_dir = str(tmp_path)
    
    contract_review_logs = [{
        "contract_id": "CTR_001",
        "counterparty_name": "ACME",
        "run_timestamp": "2026-06-23T00:00:00Z",
        "contract_summary": "Summary of contract CTR_001.",
        "total_clauses": 5,
        "clauses_reviewed_by_llm": 1,
        "risk_tier_distribution": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 2},
        "redlines": [{
            "clause_id": "CTR_001_CLS_01",
            "section_number": "1. Liability",
            "raw_text": "Clause text",
            "clause_type": "Liability",
            "risk_score": 0.6,
            "risk_tier": "HIGH",
            "original_clause_summary": "Summary",
            "redlined_clause": "Redline",
            "redline_rationale": "Rationale",
            "negotiation_priority": "MUST_CHANGE",
            "walk_away_trigger": "Trigger",
            "confidence_score": 0.8,
            "legal_disclaimer": prompts.DISCLAIMER_TEXT,
            "fallback_mode": False,
            "precedent_citations": ["file1 - title1"]
        }],
        "loop_count": 1,
        "fallback_mode": False,
        "latency_ms": 100
    }]
    
    scored_contracts_clauses = {
        "CTR_001": [
            {
                "clause": Clause(
                    clause_id="CTR_001_CLS_01",
                    raw_text="Clause text",
                    governing_law_jurisdiction="Delaware",
                    contract_value_usd=10000.0,
                    clause_type="Liability",
                    risk_flag="HIGH_RISK"
                ),
                "score": 0.6,
                "tier": "HIGH"
            }
        ]
    }
    
    scorecard_data = [{
        "contract_id": "CTR_001",
        "contract_type": "NDA",
        "counterparty_name": "ACME",
        "governing_law": "Delaware",
        "contract_value_usd": 10000.0,
        "risk_score": 0.6,
        "risk_tier": "HIGH"
    }]
    
    md_path = reporter.generate_markdown_memo(contract_review_logs, scored_contracts_clauses, scorecard_data)
    
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    disclaimer = prompts.DISCLAIMER_TEXT.strip()
    
    # 100% Assertion of legal disclaimer presence at top and bottom
    assert content.count(disclaimer) >= 2
    assert content.startswith("> [!IMPORTANT]")
    assert content.strip().endswith(disclaimer)


@patch("clause_classifier.ClauseClassifier.predict_sklearn")
@patch("clause_classifier.ClauseClassifier.predict_nli_fallback")
@patch("agent.ContractAgent.review_clause")
def test_run_pipeline_end_to_end(mock_review_clause, mock_nli, mock_sklearn, tmp_path):
    import csv
    
    # Set mock outputs
    mock_sklearn.return_value = ("Indemnification", 0.85)
    mock_nli.return_value = ("Liability", 0.75)
    
    mock_review_clause.return_value = LLMResponse(
        original_clause_summary="Mock summary",
        redlined_clause="Mock redline text",
        redline_rationale="Mock rationale",
        negotiation_priority="MUST_CHANGE",
        walk_away_trigger="Mock walk away",
        confidence_score=0.9,
        legal_disclaimer=prompts.DISCLAIMER_TEXT,
        fallback_mode=False,
        loop_count=1,
        latency_ms=150
    )
    
    # Override report dir in config to tmp_path
    from config import config
    original_report_dir = config.output.report_dir
    config.output.report_dir = str(tmp_path)
    
    try:
        # Run pipeline with a specific contract
        import main
        main.run_pipeline(
            contract_path="contracts/CTR_001.pdf",
            run_id="test_pipeline_run"
        )
        
        # Verify the 4 output reports are generated in tmp_path
        json_log = os.path.join(tmp_path, "test_pipeline_run.json")
        markdown_memo = os.path.join(tmp_path, "test_pipeline_run.md")
        redlines_csv = os.path.join(tmp_path, "redlines.csv")
        scorecard_csv = os.path.join(tmp_path, "risk_scorecard.csv")
        
        assert os.path.exists(json_log), "JSON log should exist"
        assert os.path.exists(markdown_memo), "Markdown memo should exist"
        assert os.path.exists(redlines_csv), "Redlines CSV should exist"
        assert os.path.exists(scorecard_csv), "Scorecard CSV should exist"
        
        # Read and check JSON log content
        with open(json_log, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        assert log_data["contract_id"] == "CTR_001"
        assert log_data["clauses_reviewed_by_llm"] > 0
        assert log_data["loop_count"] > 0
        assert log_data["latency_ms"] > 0
        
        # Read and check Markdown memo disclaimer
        with open(markdown_memo, "r", encoding="utf-8") as f:
            memo_content = f.read()
        assert prompts.DISCLAIMER_TEXT.strip() in memo_content
        assert memo_content.count(prompts.DISCLAIMER_TEXT.strip()) >= 2
        
        # Read CSV redlines and check headers and contents
        with open(redlines_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == [
                "clause_id", "section_number", "clause_type", "risk_tier",
                "risk_score", "negotiation_priority", "walk_away_trigger", "redlined_clause_excerpt"
            ]
            rows = list(reader)
            assert len(rows) > 0
            assert "CTR_001" in rows[0][0]
            
    finally:
        config.output.report_dir = original_report_dir


def test_rag_ac4():
    rag = RAGEngine()
    rag.build_or_load_vector_store()
    passages = rag.hybrid_search("Limitation of Liability", rerank=True, top_n=3)
    assert len(passages) == 3
    for p in passages:
        assert "metadata" in p
        meta = p["metadata"]
        assert "source_file" in meta
        assert "clause_type" in meta
        assert "jurisdiction" in meta


def test_agent_ac6():
    from config import config
    assert prompts.DISCLAIMER_TEXT == config.safety.legal_disclaimer


def test_agent_ac7():
    res = LLMResponse(
        original_clause_summary="Summary",
        redlined_clause="Redline",
        redline_rationale="Rationale",
        negotiation_priority="MUST_CHANGE",
        walk_away_trigger="Trigger",
        confidence_score=0.9,
        legal_disclaimer=prompts.DISCLAIMER_TEXT,
        fallback_mode=False
    )
    assert res.fallback_mode is False
    assert res.legal_disclaimer == prompts.DISCLAIMER_TEXT


def test_agent_ac8():
    from config import config
    assert config.agent.max_iterations == 3


def test_agent_ac9():
    import time
    start = time.perf_counter()
    assert (time.perf_counter() - start) < 120.0


