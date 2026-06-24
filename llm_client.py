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

class OllamaUsage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class OllamaTextBlock:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"

class OllamaMessageResponse:
    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0, stop_reason: str = "end_turn"):
        self.content = [OllamaTextBlock(text)]
        self.usage = OllamaUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason

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
        self.use_ollama = False
        
        from config import config
        self.local_model = config.llm.local_model if hasattr(config.llm, "local_model") else "llama3.2"
        
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            logger.warning("ANTHROPIC_API_KEY environment variable is not set. Checking for local Ollama...")
            if self._check_ollama_status():
                logger.info(f"Ollama is running locally. Falling back to local model: {self.local_model}")
                self.use_ollama = True
                self.client = None
            else:
                logger.warning("Ollama is not running locally. Real LLM calls will fail.")
                self.client = None

    def count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken (cl100k_base) as a proxy for Claude's tokenizer."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def create_message(self, **kwargs) -> Any:
        """Sends a message request to Claude, with retry logic, token guard, and cost logging."""
        # Check daily spend limit
        from config import config
        daily_limit = config.llm.daily_spend_limit
        current_daily_spend = self._get_daily_spend()
        if current_daily_spend >= daily_limit:
            logger.error(f"Daily API spend limit reached: ${current_daily_spend:.2f} >= ${daily_limit:.2f}")
            raise ValueError(f"Daily API spend limit reached: ${current_daily_spend:.2f} >= ${daily_limit:.2f}")

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
            logger.warning(f"Prompt length of {input_token_estimate} exceeds token limit of {self.token_limit}. Truncating.")
            if "messages" in kwargs and len(kwargs["messages"]) > 0:
                first_msg = kwargs["messages"][0]
                if isinstance(first_msg.get("content"), str):
                    first_msg_tokens = self.count_tokens(first_msg["content"])
                    excess = input_token_estimate - self.token_limit
                    allowed_tokens = max(0, first_msg_tokens - excess)
                    first_msg["content"] = self.truncate_text_to_tokens(first_msg["content"], allowed_tokens)
            
        if self.use_ollama:
            return self._create_ollama_message(**kwargs)

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
                
                self._add_to_daily_spend(cost)
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

    def truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncates text to a maximum number of tokens."""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)

    def _get_daily_spend(self) -> float:
        import datetime
        import json
        path = ".daily_spend.json"
        today = datetime.date.today().isoformat()
        if not os.path.exists(path):
            return 0.0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today:
                return float(data.get("spend", 0.0))
        except Exception:
            pass
        return 0.0

    def _add_to_daily_spend(self, cost: float) -> None:
        import datetime
        import json
        path = ".daily_spend.json"
        today = datetime.date.today().isoformat()
        spend = self._get_daily_spend() + cost
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"date": today, "spend": spend}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save daily spend: {e}")

    def _check_ollama_status(self) -> bool:
        import requests
        try:
            response = requests.get("http://localhost:11434", timeout=1.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        return False

    def _create_ollama_message(self, **kwargs) -> Any:
        import requests
        
        ollama_messages = []
        if "system" in kwargs and kwargs["system"]:
            ollama_messages.append({
                "role": "system",
                "content": kwargs["system"]
            })
        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                content = msg.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                    content = "\n".join(text_parts)
                ollama_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })
                
        payload = {
            "model": self.local_model,
            "messages": ollama_messages,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        max_attempts = 4
        backoff = 1.0
        
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                res_data = response.json()
                
                text = res_data.get("message", {}).get("content", "")
                input_tokens = res_data.get("prompt_eval_count", 0)
                output_tokens = res_data.get("eval_count", 0)
                
                # Keep Ollama cost to $0.00
                cost = 0.0
                
                self.cumulative_cost += cost
                self.cumulative_input_tokens += input_tokens
                self.cumulative_output_tokens += output_tokens
                
                logger.info(
                    f"Ollama Call Successful | Cost: ${cost:.6f} | "
                    f"In: {input_tokens} | Out: {output_tokens} | "
                    f"Cumulative Cost: ${self.cumulative_cost:.6f}"
                )
                
                self._add_to_daily_spend(cost)
                
                return OllamaMessageResponse(text, input_tokens, output_tokens)
                
            except Exception as e:
                logger.warning(f"Transient error on Ollama call attempt {attempt+1}/{max_attempts}: {e}")
                if attempt == max_attempts - 1:
                    logger.error("Max Ollama call attempts reached. Raising exception.")
                    raise e
                time.sleep(backoff)
                backoff *= 2.0
