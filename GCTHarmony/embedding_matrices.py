"""Embedding matrix cache (load-or-generate).

Per your design, this module is used for **one thing only**:

1) Load the background embedding matrix for a given (background, backend, mode)
2) If it doesn't exist, generate it and cache it.

All cached matrices live under the package folder:

    GCTHarmony/embeddings/

Nothing in this module is meant to be the public "query" API.

Backend / mode policy
---------------------

* If an OpenAI API key is available (via :func:`set_api_key` or environment
  variable ``OPENAI_API_KEY``):
    - backend_eff = "gpt" for both one-step & two-step.
* If no OpenAI key is available:
    - backend_eff = "qwen" for both one-step & two-step.

Mode semantics:
  - one-step: embed the label string
  - two-step: generate a 1-sentence description (GPT or local Qwen3-0.6B),
              then embed the description (GPT embedding or Qwen3 embedding)

Requested speedups implemented:
  (a) cache Qwen embedding model globally (no reload per label)
  (b) batch embedding for Qwen (and OpenAI also supports batching)
  (c) Qwen3-0.6B description: disable thinking by default + small max_new_tokens
      + truncate to first sentence
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Optional progress bar
# -----------------------------------------------------------------------------
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None  # type: ignore


# -----------------------------------------------------------------------------
# Paths / globals
# -----------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
_EMB_DIR = (_PKG_DIR / "embeddings").resolve()
_DATA_DIR = (_PKG_DIR / "data").resolve()  # ontology + HRA list live here

_OPENAI_API_KEY: Optional[str] = None


def set_api_key(api_key: Optional[str]) -> None:
    """Set OpenAI API key for the current process."""
    global _OPENAI_API_KEY
    _OPENAI_API_KEY = (api_key or "").strip() or None
    if _OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = _OPENAI_API_KEY


def has_openai_key() -> bool:
    return bool(_OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))


def choose_backend_and_mode(mode: Optional[str]) -> Tuple[str, str]:
    """Return (backend_eff, mode_eff)."""
    mode_eff = (mode or "two-step").strip().lower()
    if mode_eff not in {"one-step", "two-step"}:
        mode_eff = "two-step"
    backend_eff = "gpt" if has_openai_key() else "qwen"
    return backend_eff, mode_eff


# -----------------------------------------------------------------------------
# Background labels (CL / HRA)
# -----------------------------------------------------------------------------

_LABEL_CACHE: Dict[str, List[str]] = {}


def _load_cl_labels() -> List[str]:
    from owlready2 import get_ontology

    ontology_url = _DATA_DIR / "cl_20250129.owl"
    cl_ontology = get_ontology(str(ontology_url)).load()

    labels_all: List[str] = []
    for cls in cl_ontology.classes():
        label = cls.label
        if label and re.match(r"CL", cls.name):
            labels_all.append(str(label[0]))
    return labels_all


def _load_hra_labels_from_cl(labels_all: List[str]) -> List[str]:
    hra_path = _DATA_DIR / "HRA_cell_type.txt"
    with open(hra_path, "r", encoding="utf-8") as f:
        hra_file = [line.strip() for line in f]
    hra_lower = {x.casefold() for x in hra_file}
    hra = [lab for lab in labels_all if lab.casefold() in hra_lower]
    return sorted(hra)


def load_background_labels(background: str) -> List[str]:
    """Return label list for the requested background ("CL" or "HRA")."""
    background = background.strip().upper()
    if background not in {"CL", "HRA"}:
        raise ValueError("background must be 'CL' or 'HRA'")

    if background in _LABEL_CACHE:
        return _LABEL_CACHE[background]

    labels_all = _load_cl_labels()
    _LABEL_CACHE["CL"] = labels_all
    _LABEL_CACHE["HRA"] = _load_hra_labels_from_cl(labels_all)
    return _LABEL_CACHE[background]


# -----------------------------------------------------------------------------
# GPT backend (used only to generate matrices when backend_eff == "gpt")
# -----------------------------------------------------------------------------

def _get_openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not set. Call set_api_key(...) first.")
    return OpenAI(api_key=key)


def _gpt_embed(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    client = _get_openai_client()
    txt = text.replace("\n", " ")
    vec = client.embeddings.create(input=[txt], model=model).data[0].embedding
    return np.asarray(vec, dtype=np.float32)


def _gpt_embed_batch(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Batch OpenAI embeddings. Returns (N, D).
    """
    client = _get_openai_client()
    out: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i : i + batch_size]]
        resp = client.embeddings.create(input=batch, model=model)
        for d in resp.data:
            out.append(np.asarray(d.embedding, dtype=np.float32))

    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def _gpt_describe(cell_type: str, model: str = "gpt-4o-2024-08-06") -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Please use 1 sentence to describe cell type: {cell_type}",
            }
        ],
    )
    return resp.choices[0].message.content.strip()


# -----------------------------------------------------------------------------
# Qwen backend (used only to generate matrices when backend_eff == "qwen")
# -----------------------------------------------------------------------------

# (a) Cache Qwen embedding model globally
_QWEN_EMB_MODEL = None
_QWEN_EMB_MODEL_NAME: Optional[str] = None


def _get_qwen_emb_model(model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    """
    Lazily load and cache the SentenceTransformer model once per process.
    """
    global _QWEN_EMB_MODEL, _QWEN_EMB_MODEL_NAME

    if _QWEN_EMB_MODEL is not None and _QWEN_EMB_MODEL_NAME == model_name:
        return _QWEN_EMB_MODEL

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required for fast Qwen embedding. "
            "Install it with `pip install sentence-transformers`."
        ) from e

    _QWEN_EMB_MODEL = SentenceTransformer(model_name)
    _QWEN_EMB_MODEL_NAME = model_name
    return _QWEN_EMB_MODEL


def _qwen_embed(text: str, model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> np.ndarray:
    """Compute Qwen3 embedding for one text (fast: cached model)."""
    text = text.replace("\n", " ")
    mdl = _get_qwen_emb_model(model_name)
    vec = mdl.encode([text], normalize_embeddings=False, show_progress_bar=False)[0]
    return np.asarray(vec, dtype=np.float32)


# (b) Batch embedding for Qwen
def _qwen_embed_batch(
    texts: List[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 64,
    normalize_embeddings: bool = False,
) -> np.ndarray:
    """
    Batch Qwen3 embeddings via SentenceTransformer.encode. Returns (N, D).
    """
    mdl = _get_qwen_emb_model(model_name)
    emb = mdl.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,  # we provide tqdm outside
    )
    return np.asarray(emb, dtype=np.float32)


# Cache Qwen causal model/tokenizer globally (for two-step descriptions)
_QWEN_CAUSAL_BUNDLE = None


def _load_qwen_causal(model_name: str = "Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


# (c) Faster Qwen descriptions: disable thinking by default + short max tokens + truncate 1 sentence
def _qwen_describe(
    cell_type: str,
    *,
    enable_thinking: bool = False,
    max_new_tokens: int = 128,
) -> str:
    """Generate a 1-sentence description using local Qwen3-0.6B."""
    global _QWEN_CAUSAL_BUNDLE
    if _QWEN_CAUSAL_BUNDLE is None:
        _QWEN_CAUSAL_BUNDLE = _load_qwen_causal()
    tokenizer, model = _QWEN_CAUSAL_BUNDLE

    messages = [{"role": "user", "content": f"Please use 1 sentence to describe cell type: {cell_type}"}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Split out "thinking" if </think> token id is present (151668 in your snippet)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n").strip()

    # Enforce one sentence: cut at first sentence-ending punctuation
    for stop in [".", "!", "?"]:
        if stop in content:
            content = content.split(stop, 1)[0].strip() + stop
            break

    return content


# -----------------------------------------------------------------------------
# Embedding matrix caching / generation (ALL in GCTHarmony/embeddings/)
# -----------------------------------------------------------------------------

def _matrix_paths(background: str, backend: str, mode: str) -> Tuple[Path, Path]:
    """Return (labels_path, matrix_path) inside embeddings dir."""
    background = background.upper()
    stem = f"embeddings_{background}_{backend}_{mode}"
    _EMB_DIR.mkdir(parents=True, exist_ok=True)
    return _EMB_DIR / f"{stem}__labels.npy", _EMB_DIR / f"{stem}.npy"


def get_background_embeddings(
    background: str = "HRA",
    mode: str = "two-step",
    force_rebuild: bool = False,
    *,
    # batch knobs
    qwen_embed_batch_size: int = 64,
    openai_embed_batch_size: int = 128,
) -> Tuple[str, str, List[str], np.ndarray]:
    """Load (or generate) the background embedding matrix.

    Returns:
        backend_eff, mode_eff, labels, emb_matrix
    """
    backend_eff, mode_eff = choose_backend_and_mode(mode)
    background_u = background.strip().upper()
    if background_u not in {"CL", "HRA"}:
        raise ValueError("background must be 'CL' or 'HRA'")

    # 1) labels
    labels = load_background_labels(background_u)

    # 2) cache load
    labels_path, mat_path = _matrix_paths(background_u, backend_eff, mode_eff)
    if (not force_rebuild) and mat_path.exists() and labels_path.exists():
        emb = np.load(mat_path)
        return backend_eff, mode_eff, labels, emb

    # 3) cache miss -> build
    print(
        f"[GCTHarmony] Building embedding matrix "
        f"(background={background_u}, backend={backend_eff}, mode={mode_eff}, N={len(labels)})"
    )

    # 3a) build texts to embed
    if mode_eff == "one-step":
        texts = labels
    else:
        it = labels
        if tqdm is not None:
            it = tqdm(labels, desc=f"[GCTHarmony] Describing {background_u}", unit="cell-type")

        if backend_eff == "gpt":
            texts = [_gpt_describe(lab) for lab in it]
        else:
            texts = [_qwen_describe(lab) for lab in it]

    # 3b) embed texts (batched)
    n = len(texts)

    if backend_eff == "gpt":
        # Optional batch progress for OpenAI
        if tqdm is not None:
            vecs: List[np.ndarray] = []
            batch_iter = tqdm(
                range(0, n, openai_embed_batch_size),
                desc=f"[GCTHarmony] Embedding {background_u} (OpenAI)",
                unit="batch",
            )
            for i in batch_iter:
                vecs.append(_gpt_embed_batch(texts[i : i + openai_embed_batch_size], batch_size=openai_embed_batch_size))
            emb = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
        else:
            emb = _gpt_embed_batch(texts, batch_size=openai_embed_batch_size)
    else:
        # Qwen embedding: chunked batching so tqdm advances smoothly
        if tqdm is not None:
            vecs = []
            batch_iter = tqdm(
                range(0, n, qwen_embed_batch_size),
                desc=f"[GCTHarmony] Embedding {background_u} (Qwen)",
                unit="batch",
            )
            for i in batch_iter:
                vecs.append(_qwen_embed_batch(texts[i : i + qwen_embed_batch_size], batch_size=qwen_embed_batch_size))
            emb = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
        else:
            emb = _qwen_embed_batch(texts, batch_size=qwen_embed_batch_size)

    emb = np.asarray(emb, dtype=np.float32)

    # 4) cache
    np.save(labels_path, np.asarray(labels, dtype=object))
    np.save(mat_path, emb)
    print(f"[GCTHarmony] Saved embedding matrix: {mat_path}")
    return backend_eff, mode_eff, labels, emb


# NOTE: This module intentionally does NOT expose query-time helpers like
# "embed_query" or "describe_cell_type". Keep query logic in first_module.py.
