# GCTHarmony

**GCTHarmony** is a Python package for harmonizing cell type annotations across single-cell and spatial transcriptomics datasets by mapping heterogeneous, user-defined labels to standardized **Cell Ontology (CL)** terms in a robust and reproducible way.

The package integrates **LLM-based semantic reasoning** with **embedding-based similarity matching**, and supports automatic backend switching to ensure full functionality even when external APIs are unavailable.

---

## Key Features

- **Automatic dual backend**
  - **OpenAI available** → GPT-based description + embedding
  - **No OpenAI key** → fully local **Qwen** backend
- **Two mapping strategies**
  - **One-step**: directly embed input labels
  - **Two-step**: generate a short semantic description, then embed
- **Ontology-aware background choices**
  - **HRA**: commonly used / curated cell types (faster, lower noise)
  - **Full CL**: complete Cell Ontology coverage
- **Embedding matrix caching**
  - Background embeddings cached as `.npy` files for reproducibility and speed
- **Hierarchy reconciliation**
  - Resolve granularity mismatches between related ontology terms

---

## Installation

### Local Development Mode (Recommended)

```bash
# Clone the repository
git clone git@github.com:Xingyuan-Zhang/GCTHarmony.git
cd GCTHarmony
# Install in editable mode
pip install -e .
```

### Backend Configuration

GCTHarmony automatically selects the backend at runtime based on the availability of an OpenAI API key. No user intervention is required in most cases.

## OpenAI backend (optional)

If an OpenAI API key is available, GCTHarmony will use GPT-based models for both semantic description generation and embedding.

Set the key via environment variable:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Or directly in Python:

```python
from GCTHarmony.first_module import set_api_key
set_api_key("YOUR_API_KEY")
```
When the OpenAI backend is active:

One-step strategy: embeddings are generated directly from the input label

Two-step strategy: a one-sentence semantic description is generated first, then embedded

If no OpenAI API key is detected, GCTHarmony automatically switches to the Qwen backend, enabling fully local execution.

## Module 1: Cell Type Harmonization
This module maps raw cell type strings to standardized CL terms using semantic similarity.

### Basic usage
```python
from GCTHarmony.first_module import GCTHarmony

labels = ["T-cells", "AT1", "B cell"]

mapped = GCTHarmony(
    labels,
    mode="two-step",      # "one-step" or "two-step"
    background="HRA"      # "HRA" or "CL"
)

print(mapped)
```
Example output:
```python
['T cell', 'alveolar type I pneumocyte', 'B cell']
```
Parameter	Description
labels:	List of raw cell type strings
mode:	"one-step" or "two-step"
background:	"HRA" (default) or "CL"

## Module 2: Hierarchy Reconciliation

```python
from GCTHarmony.second_module import merge_cell_types

a, b = merge_cell_types(
    ["CD4-positive, alpha-beta T cell"],
    ["T cell"]
)

print(a, b)
```

## Side function: map to a user-provided standard label set

In addition to mapping to CL/HRA backgrounds, GCTHarmony provides a lightweight **side function** for the following use case:

> You already have a **fixed list of standard labels** (e.g., your lab’s canonical annotation set).  
> You want to map any new/free-text label to **the closest label in that standard set**.

### What it does

1. Takes a list of strings as **standard labels**
2. Builds (or loads) an **embedding matrix** for those standard labels (cached as `.npy`)
3. Given any new text label(s), returns the **closest standard label** by cosine similarity

### Example

```python
# Example name; use the actual function name in your package if different.
from GCTHarmony.side_functions import map_to_standard_labels

standard_labels = [
    "T cell",
    "B cell",
    "macrophage",
    "endothelial cell",
    "fibroblast",
]

queries = ["T-cells", "B lymphocyte", "endo cells"]

mapped = map_to_standard_labels(
    queries=queries,
    standard_labels=standard_labels,
    mode="one-step",          # or "two-step"
    cache_index="my_lab_v1"   # used to save/load the .npy cache consistently
)

print(mapped)  # dict: {query -> closest_standard_label}
```

Example output:

```text
{
  "T-cells": "T cell",
  "B lymphocyte": "B cell",
  "endo cells": "endothelial cell"
}
```

### Notes

- `cache_index` should be a stable identifier (e.g. `"my_lab_v1"`) so you can reuse the same cached embedding matrix across runs.
- The mapping is purely embedding-similarity based (cosine similarity) and is designed for fast harmonization to a fixed label set.

---

## Embedding Cache

Background embedding matrices are automatically cached to disk.

Default cache location:

```
GCTHarmony/embeddings/

---

## Authors

- **Xingyuan Zhang** — Duke University  
- **Zhicheng Ji** — Duke University  

---

## License

MIT License




