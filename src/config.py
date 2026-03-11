"""
Configuration management for the multi-agent debate system.
Reads from environment variables with sensible defaults for local Ollama.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the debate system."""
    
    # Model configuration
    model: str = "llama3.2:3b"
    api_base_url: str = "http://localhost:11434"
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 120.0
    
    # Experiment tracking
    phase: str = "phase1"
    
    # Optional API keys (for Phase 4+)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config instance from environment variables."""
        return cls(
            # Model configuration
            model=os.environ.get("MODEL", "llama3.2:3b"),
            api_base_url=os.environ.get("API_BASE_URL", "http://localhost:11434"),
            
            # Generation parameters
            temperature=float(os.environ.get("TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            timeout=float(os.environ.get("TIMEOUT", "30.0")),
            
            # Experiment tracking
            phase=os.environ.get("PHASE", "phase1"),
            
            # API keys (optional)
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
    
    def safe_summary(self) -> str:
        """
        Return a human-readable summary of the configuration with API keys redacted.
        Safe for logging without exposing secrets.
        """
        lines = [
            "Configuration Summary:",
            f"  Model: {self.model}",
            f"  API Base URL: {self.api_base_url}",
            f"  Temperature: {self.temperature}",
            f"  Max Tokens: {self.max_tokens}",
            f"  Timeout: {self.timeout}s",
            f"  Phase: {self.phase}",
            f"  Anthropic API Key: {self._redact_key(self.anthropic_api_key)}",
            f"  OpenAI API Key: {self._redact_key(self.openai_api_key)}",
        ]
        return "\n".join(lines)
    
    def _redact_key(self, key: Optional[str]) -> str:
        """Redact an API key for safe logging."""
        if not key:
            return "[not set]"
        if len(key) <= 8:
            return "********"  # Too short to show anything
        # Show first 4 and last 4 characters
        return f"{key[:4]}...{key[-4:]}"
    
    def __str__(self) -> str:
        """
        String representation (uses safe_summary to avoid exposing keys).
        This makes print(config) safe by default.
        """
        return self.safe_summary()


# Create a global config instance for easy import
config = Config.from_env()