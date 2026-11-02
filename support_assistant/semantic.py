from functools import lru_cache
import os
from typing import Any, Sequence

import numpy as np

from support_assistant.config import ROOT


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL = ROOT / "models" / "all-MiniLM-L6-v2"


class SemanticModelUnavailable(RuntimeError):
    pass


def semantic_model_source() -> str | None:
    configured = os.getenv("SENTENCE_TRANSFORMER_MODEL", "").strip()
    if configured:
        return configured
    return str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else None


@lru_cache(maxsize=2)
def load_sentence_model(source: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SemanticModelUnavailable(
            "Sentence Transformers is not installed; install the project requirements to enable semantic search."
        ) from error
    try:
        return SentenceTransformer(source, device="cpu")
    except Exception as error:
        raise SemanticModelUnavailable(
            "The configured Sentence Transformer model could not be loaded. Check SENTENCE_TRANSFORMER_MODEL and the model files."
        ) from error


def encode_sentences(encoder: Any, texts: Sequence[str]) -> np.ndarray:
    try:
        vectors = np.asarray(encoder.encode(
            list(texts), batch_size=32, convert_to_numpy=True, show_progress_bar=False,
        ), dtype=np.float32)
    except Exception as error:
        raise SemanticModelUnavailable("Sentence Transformer encoding failed.") from error
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise SemanticModelUnavailable("Sentence Transformer returned invalid embeddings.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def download_model() -> int:
    """Fetch the model once so the app can run semantic search offline from models/."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_NAME, device="cpu").save(str(LOCAL_MODEL))
    print(f"Saved {MODEL_NAME} to {LOCAL_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(download_model())