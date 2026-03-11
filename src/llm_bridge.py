"""
Bridge between JARVIS and the MiniTransformer language model.
"""

import sys
import os
from typing import Optional, Any, Dict

# Build absolute path to LLM directory
JARVIS_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(JARVIS_DIR, '..', 'LLM')
MODEL_PATH = os.path.join(LLM_DIR, 'models', 'mini_transformer_best.pt')

# Add LLM directory to Python path
sys.path.insert(0, os.path.abspath(LLM_DIR))

# Try to import from LLM directory
try:
    from generate_transformer import load_transformer, generate as transformer_generate  # type: ignore
    from mini_transformer import MiniTransformer  # type: ignore
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
   
# Global model cache
_model: Optional[Any] = None
_stoi: Optional[Dict] = None
_itos: Optional[Dict] = None
_block_size: Optional[int] = None


def initialize_llm():
    """Load the transformer model once and cache it"""
    global _model, _stoi, _itos, _block_size
    
    if not LLM_AVAILABLE:
        print("LLM imports not available")
        return False
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return False
    
    if _model is None:
        try:
            print("Loading transformer model...")
            _model, _stoi, _itos, _block_size = load_transformer(MODEL_PATH)
            print("✓ Transformer ready")
            return True
        except Exception as e:
            print(f"Error loading transformer: {e}")
            return False
    
    return True


def generate_text(prompt, length=200, temperature=0.7):
    """Generate text using the transformer"""
    if not LLM_AVAILABLE:
        return None
    
    if _model is None:
        if not initialize_llm():
            return None
    
    try:
        text = transformer_generate(
            _model, _stoi, _itos,
            prompt, length, temperature, _block_size
        )
        return text
    except Exception as e:
        print(f"Error generating text: {e}")
        return None


def is_available():
    """Check if LLM is available"""
    return LLM_AVAILABLE and os.path.exists(MODEL_PATH)