"""First module: map raw cell-type strings to CL/HRA terms.

Public API preserved from the original package:
  - set_api_key
  - find_closest_cell_type
  - get_response
  - GCTHarmony

Implementation changes (requested):
  - If OpenAI key is available -> use GPT for both one-step & two-step.
  - If OpenAI key is NOT available:
        * one-step uses Qwen3 Embedding
        * two-step uses Qwen3-0.6B (local) to generate a description, then
          Qwen3 Embedding

Embedding matrices are loaded/generated via :mod:`GCTHarmony.embedding_matrices`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import os

import numpy as np
from scipy.spatial.distance import cosine

from .embedding_matrices import (
    set_api_key,  # re-export
    choose_backend_and_mode,
    get_background_embeddings,
)


# -----------------------------------------------------------------------------
# Query-time helpers (kept here; embedding_matrices.py is only for matrices)
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


def _qwen_embed(text: str, model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> np.ndarray:
    """Compute Qwen3 embedding.

    Tries sentence-transformers first, otherwise falls back to a minimal
    Transformers mean-pooling implementation.
    """
    text = text.replace("\n", " ")

    try:
        from sentence_transformers import SentenceTransformer

        mdl = SentenceTransformer(model_name)
        vec = mdl.encode([text], normalize_embeddings=False)[0]
        return np.asarray(vec, dtype=np.float32)
    except Exception:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval()

        with torch.no_grad():
            batch = tok([text], padding=True, truncation=True, return_tensors="pt")
            out = mdl(**batch)
            hidden = out.last_hidden_state  # (B, T, H)
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vec = pooled[0].cpu().numpy()
        return np.asarray(vec, dtype=np.float32)


_QWEN_CAUSAL_BUNDLE = None


def _load_qwen_causal(model_name: str = "Qwen/Qwen3-0.6B"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def _qwen_describe(cell_type: str, *, enable_thinking: bool = True, max_new_tokens: int = 256) -> str:
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
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Split out "thinking" if </think> token id is present (151668 in your snippet)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n").strip()
    return content


def embed_query(text: str, mode: str) -> np.ndarray:
    """Embed an arbitrary query string using the effective backend."""
    backend_eff, _ = choose_backend_and_mode(mode)
    if backend_eff == "gpt":
        return _gpt_embed(text)
    return _qwen_embed(text)


def describe_cell_type(ct: str) -> str:
    """1-sentence description (GPT if key, else Qwen3-0.6B local)."""
    backend_eff, _ = choose_backend_and_mode("two-step")
    if backend_eff == "gpt":
        return _gpt_describe(ct)
    return _qwen_describe(ct)


def find_closest_cell_type(
    name: str,
    mode: str = "two-step",
    background: str = "CL",
) -> Tuple[str, float]:
    """Find the closest background term to the given query string."""
    _, mode_eff = choose_backend_and_mode(mode)
    _, _, labels, emb_matrix = get_background_embeddings(background=background, mode=mode_eff)

    target = embed_query(name, mode_eff)

    closest_name = labels[0]
    closest_distance = float("inf")
    for cell_type, embedding in zip(labels, emb_matrix):
        d = float(cosine(target, embedding))
        if d < closest_distance:
            closest_distance = d
            closest_name = cell_type
    return closest_name, closest_distance


def get_response(prompt: str) -> str:
    """Return a 1-sentence description of a cell type."""
    return describe_cell_type(prompt)


from typing import Dict, List, Optional, Union

def GCTHarmony(
    input_cts: Union[str, List[str]],
    mode: str = "two-step",
    background: str = "HRA",
    aux_info: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    
    aux_info = aux_info or {}
    _, mode_eff = choose_backend_and_mode(mode)

    if isinstance(input_cts, str):
        input_list = [input_cts]
    else:
        input_list = list(input_cts)

    out: Dict[str, str] = {}
    for ct in input_list:
        aux = aux_info.get(ct)

        if mode_eff == "one-step":
            query_text = ct if not aux else f"{ct} ({aux})"
        else:
            descr = get_response(ct)
            query_text = descr if not aux else f"{descr} ({aux})"

        mapped, _ = find_closest_cell_type(
            query_text,
            mode=mode_eff,
            background=background,
        )
        out[ct] = mapped

    return out



# -----------------------------------------------------------------------------
# Side utility: map to a user-provided standard label set (ONE-STEP ONLY)
#   - no aux_info
#   - no extra text normalization beyond your embed_query(...)
#   - cache standard embedding matrices under ./embedding/custom/
# -----------------------------------------------------------------------------

import os
import json

import pandas as pd
from tqdm import tqdm



def _custom_cache_dir() -> str:
    d = os.path.join(".", "embedding", "custom")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_paths(index_name: str) -> Dict[str, str]:
    base = os.path.join(_custom_cache_dir(), index_name)
    return {
        "npy": base + ".npy",
        "labels": base + ".labels.json",
    }


def _save_index_npy(emb: np.ndarray, labels: List[str], index_name: str) -> None:
    paths = _cache_paths(index_name)
    np.save(paths["npy"], emb.astype(np.float32, copy=False))
    with open(paths["labels"], "w") as f:
        json.dump(labels, f, indent=2)


def _load_index_npy(index_name: str) -> tuple[List[str], np.ndarray]:
    paths = _cache_paths(index_name)
    if not os.path.exists(paths["npy"]) or not os.path.exists(paths["labels"]):
        raise FileNotFoundError(
            f"Custom index '{index_name}' not found. Expected:\n"
            f"  - {paths['npy']}\n"
            f"  - {paths['labels']}"
        )

    emb = np.load(paths["npy"]).astype(np.float32, copy=False)
    with open(paths["labels"], "r") as f:
        labels = json.load(f)

    if emb.shape[0] != len(labels):
        raise ValueError(
            f"Cache mismatch for '{index_name}': emb has {emb.shape[0]} rows "
            f"but labels has {len(labels)} items."
        )

    return labels, emb


def build_standard_label_index(
    standard_labels: List[str],
    index_name: str,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build (or load) an embedding matrix for a user-provided standard label vocabulary.

    Cache:
      ./embedding/custom/{index_name}.npy
      ./embedding/custom/{index_name}.labels.json

    Returns
    -------
    pd.DataFrame
        Rows correspond to standard labels; columns are embedding dimensions.
    """
    paths = _cache_paths(index_name)
    if (not overwrite) and os.path.exists(paths["npy"]) and os.path.exists(paths["labels"]):
        labels, emb = _load_index_npy(index_name)
        d = emb.shape[1]
        return pd.DataFrame(emb, index=labels, columns=[f"dim_{i}" for i in range(d)])

    iterator = standard_labels
    if show_progress:
        iterator = tqdm(standard_labels, desc=f"Embedding standard labels [{index_name}]", unit="label")

    vecs = []
    for lab in iterator:
        v = embed_query(lab, "one-step")  # GPT embedding if key else Qwen embedding
        vecs.append(np.asarray(v, dtype=np.float32))

    emb = np.vstack(vecs)  # (n_labels, dim)
    _save_index_npy(emb, list(standard_labels), index_name)

    d = emb.shape[1]
    return pd.DataFrame(emb, index=list(standard_labels), columns=[f"dim_{i}" for i in range(d)])


def map_text_to_standard_labels(
    custom_labels: Union[str, List[str]],
    index_name: str,
) -> Dict[str, str]:
    """
    Map custom label(s) to the closest label in a cached standard label index.

    Input:
      - custom_labels: str or list[str]
      - index_name: must be the same string used in build_standard_label_index(...)

    Return:
      - dict {custom_label -> best_standard_label}
    """
    if isinstance(custom_labels, str):
        query_list = [custom_labels]
    else:
        query_list = list(custom_labels)

    std_labels, std_mat = _load_index_npy(index_name)          # (n_std, dim)
    std_mat = std_mat.astype(np.float32, copy=False)

    # cosine similarity via normalized dot product
    std_norm = np.linalg.norm(std_mat, axis=1)
    std_norm = np.clip(std_norm, 1e-12, None)

    out: Dict[str, str] = {}
    for q in query_list:
        qv = np.asarray(embed_query(q, "one-step"), dtype=np.float32)
        qn = float(np.linalg.norm(qv))
        qn = max(qn, 1e-12)

        sims = (std_mat @ qv) / (std_norm * qn)   # (n_std,)
        best_idx = int(np.argmax(sims))
        out[q] = std_labels[best_idx]

    return out