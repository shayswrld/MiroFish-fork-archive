"""
Embeddings utility — uses Ollama /v1/embeddings or OpenAI-compatible endpoint.
Falls back to None if embeddings endpoint unavailable.
"""

import json
from typing import List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.embeddings")

_embedding_client = None


def _get_client():
    global _embedding_client
    if _embedding_client is None:
        from openai import OpenAI

        _embedding_client = OpenAI(
            api_key=Config.LLM_API_KEY or "ollama",
            base_url=Config.LLM_BASE_URL,
        )
    return _embedding_client


def embed(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """Embed text via the configured LLM endpoint. Returns None on failure."""
    if not text or not text.strip():
        return None
    embed_model = model or Config.LLM_MODEL_NAME
    try:
        client = _get_client()
        resp = client.embeddings.create(model=embed_model, input=text[:8000])
        if resp.data and len(resp.data) > 0:
            return resp.data[0].embedding
    except Exception as e:
        logger.debug(
            f"Embedding failed (non-fatal, keyword search will be used): {str(e)[:100]}"
        )
    return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity. ponytail: O(d) per comparison, fine for small graphs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
