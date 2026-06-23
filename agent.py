import os
import re
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger

from models import Clause
from llm_client import LLMClient
from rag_engine import RAGEngine
from risk_scorer import RiskScorer
import prompts

class RedlineResponse(BaseModel):
    """Structured legal redline suggestion response."""
    disclaimer: str = Field(..., description="The mandatory legal disclaimer exact string.")
    suggested_redline: str = Field(..., description="The suggested revised text for the clause.")
    explanation: str = Field(..., description="Legal reasoning and citation details.")

class ContractAgent:
    """Legal Agent that reviews contract clauses using a ReAct loop with tools and structured responses."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_engine: Optional[RAGEngine] = None,
        risk_scorer: Optional[RiskScorer] = None
    ):
        self.llm_client = llm_client or LLMClient()
        self.rag_engine = rag_engine or RAGEngine()
        
        # Load risk scorer (ensuring it is initialized)
        self.risk_scorer = risk_scorer or RiskScorer(rag_engine=self.rag_engine)
        
        # Ensure RAG knowledge base is loaded
        self.rag_engine.build_or_load_vector_store()
        
        # Define native Claude tools
        self.tools = [
            {
                "name": "precedent_lookup",
                "description": "Find relevant market precedents and guidelines in the knowledge base.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., clause text, concept, or keywords)."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "jurisdiction_lookup",
                "description": "Retrieve the legal risk score for a given jurisdiction/state.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "jurisdiction": {
                            "type": "string",
                            "description": "The name of the state or jurisdiction (e.g. California, Delaware)."
                        }
                    },
                    "required": ["jurisdiction"]
                }
            }
        ]

    def _precedent_lookup_tool(self, query: str) -> str:
        results = self.rag_engine.hybrid_search(query, rerank=True, top_n=3)
        if not results:
            return "No precedents found."
        
        formatted = []
        for idx, r in enumerate(results):
            meta = r.get("metadata", {})
            formatted.append(
                f"Precedent {idx+1}:\n"
                f"Source File: {meta.get('source_file')}\n"
                f"Title: {meta.get('title')}\n"
                f"Jurisdiction: {meta.get('jurisdiction')}\n"
                f"Category: {meta.get('clause_type')}\n"
                f"Content: {r.get('text')}\n"
            )
        return "\n---\n".join(formatted)

    def _jurisdiction_lookup_tool(self, jurisdiction: str) -> str:
        score = self.risk_scorer.lookup_jurisdiction_risk(jurisdiction)
        return f"Jurisdiction '{jurisdiction}' risk score: {score}"

    def review_clause(self, clause: Clause) -> RedlineResponse:
        """Runs the ReAct loop to generate a structured redline for a clause, with RAG fallback."""
        # 1. Fetch initial RAG precedents context for the user prompt
        precedents = self.rag_engine.hybrid_search(clause.raw_text, rerank=True, top_n=3)
        precedent_context = self._format_precedent_context(precedents)
        
        # Calculate individual clause risk score
        clause_res = self.risk_scorer.score_clause(clause, precedents)
        clause_risk_score = clause_res["final_score"]
        
        # 2. Build User Prompt
        user_prompt = prompts.USER_PROMPT_TEMPLATE.format(
            clause_id=clause.clause_id,
            clause_type=clause.clause_type or "Unknown",
            raw_text=clause.raw_text,
            governing_law=clause.governing_law_jurisdiction or "Unknown",
            contract_value_usd=clause.contract_value_usd or 0.0,
            counterparty_name=clause.counterparty_name or "Unknown",
            risk_score=round(clause_risk_score, 4),
            precedent_context=precedent_context
        )
        
        # 3. Execute ReAct Loop (with fallback)
        try:
            final_text = self._run_react_loop(clause, user_prompt)
            response_data = self._parse_and_validate(final_text)
            return response_data
        except Exception as e:
            logger.warning(
                f"ReAct loop or response validation failed for clause {clause.clause_id}: {e}. "
                f"Triggering RAG fallback."
            )
            return self._rag_fallback(clause)

    def _format_precedent_context(self, precedents: List[dict]) -> str:
        if not precedents:
            return "No precedents found."
        formatted = []
        for idx, p in enumerate(precedents):
            meta = p.get("metadata", {})
            formatted.append(
                f"- Precedent {idx+1} from {meta.get('source_file')}: \"{p.get('text')[:300]}...\""
            )
        return "\n".join(formatted)

    def _run_react_loop(self, clause: Clause, user_prompt: str) -> str:
        """Executes the tool-calling loop with Claude."""
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        max_turns = 5
        for turn in range(max_turns):
            logger.info(f"ReAct Loop Turn {turn+1}/{max_turns} for clause {clause.clause_id}")
            
            # Send message to Claude
            response = self.llm_client.create_message(
                system=prompts.SYSTEM_PROMPT,
                messages=messages,
                tools=self.tools
            )
            
            # Extract text blocks and tool use blocks
            text_blocks = []
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)
            
            # Append assistant response to messages history
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            if not tool_calls:
                # Claude has finished reasoning and returned final text
                final_text = "\n".join(text_blocks)
                return final_text
                
            # Execute tool requests
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_id = tool_call.id
                tool_args = tool_call.input
                
                logger.info(f"Agent executing tool: {tool_name} with args: {tool_args}")
                
                try:
                    if tool_name == "precedent_lookup":
                        query = tool_args.get("query", "")
                        result = self._precedent_lookup_tool(query)
                    elif tool_name == "jurisdiction_lookup":
                        jurisdiction = tool_args.get("jurisdiction", "")
                        result = self._jurisdiction_lookup_tool(jurisdiction)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"
                except Exception as e:
                    logger.error(f"Failed to execute tool {tool_name}: {e}")
                    result = f"Error during execution: {str(e)}"
                    
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result
                })
                
            # Append tool results as a single user message
            messages.append({
                "role": "user",
                "content": tool_results
            })
            
        raise TimeoutError("Max ReAct loop turns reached without final answer.")

    def _parse_and_validate(self, text: str) -> RedlineResponse:
        """Parses LLM response JSON block and validates exact-string disclaimer match."""
        # Find JSON boundaries
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = text
            
        data = json.loads(json_str)
        response_model = RedlineResponse(**data)
        
        # Exact-string validation
        if response_model.disclaimer.strip() != prompts.DISCLAIMER_TEXT.strip():
            logger.error("Disclaimer text validation failed.")
            raise ValueError(
                f"Disclaimer text mismatch.\n"
                f"Expected: '{prompts.DISCLAIMER_TEXT}'\n"
                f"Got: '{response_model.disclaimer}'"
            )
            
        return response_model

    def _rag_fallback(self, clause: Clause) -> RedlineResponse:
        """Fallback mechanism if LLM fails, times out, or fails disclaimer validation."""
        logger.warning(f"Applying RAG Fallback suggestion for clause {clause.clause_id}")
        
        # Retrieve the best precedent passage
        precedents = self.rag_engine.hybrid_search(clause.raw_text, rerank=True, top_n=1)
        
        if precedents:
            suggested_redline = precedents[0]["text"]
            source = precedents[0]["metadata"].get("source_file", "Unknown")
            title = precedents[0]["metadata"].get("title", "Standard Playbook")
            explanation = (
                f"RAG Fallback suggestions: Proposing revision based on market precedent "
                f"from {source} ({title}) due to an LLM timeout or structured parsing exception."
            )
        else:
            suggested_redline = clause.raw_text
            explanation = (
                "RAG Fallback suggestions: Proposing original clause text as no precedents "
                "were found in the knowledge base."
            )
            
        return RedlineResponse(
            disclaimer=prompts.DISCLAIMER_TEXT,
            suggested_redline=suggested_redline,
            explanation=explanation
        )

    def review_contract(self, filepath: str) -> List[Dict[str, Any]]:
        """Reviews a contract: parses clauses, scores risks, and runs ReAct loop for HIGH_RISK ones."""
        logger.info(f"Reviewing contract: {filepath}")
        
        # Import parser helper
        from contract_parser import parse_contract
        
        clauses = parse_contract(filepath)
        if not clauses:
            logger.warning("No clauses parsed for contract review.")
            return []
            
        # Compute risk scores to sort clauses
        scored_clauses = []
        for clause in clauses:
            # Get precedents (needed for market deviation score)
            precedents = []
            if clause.risk_flag in ["HIGH_RISK", "REVIEW_REQUIRED"]:
                precedents = self.rag_engine.hybrid_search(clause.raw_text, rerank=True, top_n=3)
            
            scored_res = self.risk_scorer.score_clause(clause, precedents)
            scored_clauses.append((clause, scored_res["final_score"]))
            
        # Sort clauses by risk score descending
        scored_clauses = sorted(scored_clauses, key=lambda x: x[1], reverse=True)
        
        redlines = []
        for clause, score in scored_clauses:
            # We review and generate suggestions only for HIGH_RISK clauses
            if clause.risk_flag == "HIGH_RISK":
                logger.info(f"Reviewing HIGH_RISK clause {clause.clause_id} (Score={score:.4f})...")
                res = self.review_clause(clause)
                redlines.append({
                    "clause_id": clause.clause_id,
                    "section_number": clause.section_number,
                    "raw_text": clause.raw_text,
                    "clause_type": clause.clause_type,
                    "risk_score": score,
                    "disclaimer": res.disclaimer,
                    "suggested_redline": res.suggested_redline,
                    "explanation": res.explanation
                })
            else:
                logger.info(f"Skipping clause {clause.clause_id} with risk flag {clause.risk_flag}")
                
        return redlines
