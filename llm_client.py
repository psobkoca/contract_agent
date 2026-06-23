import os
import time
import tiktoken
from typing import Optional, Dict, Any
from loguru import logger

# Import anthropic client and exception classes
import anthropic
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    APIStatusError
)

class LLMClient:
    """Claude SDK wrapper with 3-retry, token guard, and cost logging."""
    
    # Model pricing per token
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "claude-3-5-sonnet-20241022": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "default": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000}
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        token_limit: int = 2000
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.token_limit = token_limit
        
        self.cumulative_cost = 0.0
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            logger.warning("ANTHROPIC_API_KEY environment variable is not set. Real LLM calls will fail.")
            self.client = None

    def count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken (cl100k_base) as a proxy for Claude's tokenizer."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def create_message(self, **kwargs) -> Any:
        """Sends a message request to Claude, with retry logic, token guard, and cost logging."""
        # 1. Token Guard check
        # Sum up system prompt and messages content
        total_input_chars = 0
        if "system" in kwargs and isinstance(kwargs["system"], str):
            total_input_chars += len(kwargs["system"])
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            for msg in kwargs["messages"]:
                if "content" in msg:
                    if isinstance(msg["content"], str):
                        total_input_chars += len(msg["content"])
                    elif isinstance(msg["content"], list):
                        for block in msg["content"]:
                            if isinstance(block, dict) and "text" in block:
                                total_input_chars += len(block["text"])
                                
        # Count tokens on the approximate input text
        # Since tiktoken is precise, we reconstruct the raw prompt text for token guard
        reconstructed_text = ""
        if "system" in kwargs and kwargs["system"]:
            reconstructed_text += kwargs["system"] + "\n"
        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                if "content" in msg:
                    if isinstance(msg["content"], str):
                        reconstructed_text += msg["content"] + "\n"
                    elif isinstance(msg["content"], list):
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                reconstructed_text += block.get("text", "") + "\n"
                                
        input_token_estimate = self.count_tokens(reconstructed_text)
        
        if input_token_estimate > self.token_limit:
            logger.error(f"Token Guard Triggered: input is {input_token_estimate} tokens, limit is {self.token_limit}")
            raise ValueError(f"Prompt length of {input_token_estimate} tokens exceeds token guard limit of {self.token_limit}")
            
        if not self.client:
            raise RuntimeError("Anthropic client is not initialized (missing API key).")
            
        # Ensure model is set
        if "model" not in kwargs:
            kwargs["model"] = self.model
            
        # 2. Retry Mechanism (3 retries = 4 attempts max)
        max_attempts = 4
        backoff = 1.0
        
        for attempt in range(max_attempts):
            try:
                response = self.client.messages.create(**kwargs)
                
                # 3. Cost Logging
                usage = response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                
                pricing_rates = self.PRICING.get(kwargs["model"], self.PRICING["default"])
                cost = (input_tokens * pricing_rates["input"]) + (output_tokens * pricing_rates["output"])
                
                self.cumulative_cost += cost
                self.cumulative_input_tokens += input_tokens
                self.cumulative_output_tokens += output_tokens
                
                logger.info(
                    f"LLM Call Successful | Cost: ${cost:.6f} | "
                    f"In: {input_tokens} | Out: {output_tokens} | "
                    f"Cumulative Cost: ${self.cumulative_cost:.6f}"
                )
                
                return response
                
            except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
                logger.warning(f"Transient error on LLM call attempt {attempt+1}/{max_attempts}: {e}")
                if attempt == max_attempts - 1:
                    logger.error("Max LLM call attempts reached. Raising exception.")
                    raise e
                time.sleep(backoff)
                backoff *= 2.0
                
            except APIStatusError as e:
                # Retry on 5xx status codes
                if e.status_code >= 500:
                    logger.warning(f"Server error {e.status_code} on LLM call attempt {attempt+1}/{max_attempts}: {e}")
                    if attempt == max_attempts - 1:
                        logger.error("Max LLM call attempts reached. Raising exception.")
                        raise e
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(f"Non-retryable API status error {e.status_code}: {e}")
                    raise e
