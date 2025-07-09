# GCTHarmony

**GCTHarmony** is a Python package designed to harmonize cell type annotations across single-cell studies by mapping original labels to standardized Cell Ontology (CL) terms, improving consistency and comparability between datasets.

## Installation

### From PyPI (when released)

```bash
pip install GCTHarmony
```

### From GitHub (development version)

```bash
pip install git@github.com:Xingyuan-Zhang/GCTHarmony.git
```

### Local Development Mode

```bash
# Clone the repository
git clone git@github.com:Xingyuan-Zhang/GCTHarmony.git
cd GCTHarmony
# Install in editable mode
pip install -e .
```

---

## Quick Start

GCTHarmony is not a free package to be used. An OpanAI API key is needed for running GCTHarmony first module. Setup the API key first before calling the first module of GCTHarmony. 

```bash
>>> from GCTHarmony.first_module import set_api_key
>>> set_api_key('Your OpenAI API Key')
```
First module: Mapping cell type names to CL terms
Example: 
```bash
>>> from GCTHarmony.first_module import GCTHarmony
>>> GCTHarmony(['T-cells'], mode='two-step', background='HRA')
['T cell']
```

Second module: reconcilling hierarchical discrepancies
Example: 
```bash
>>> from GCTHarmony.second_module import merge_cell_types
>>> merge_cell_types(['CD4-positive, alpha-beta T cell'],['T cell'])
(['T cell'], ['T cell'])
```
---
## Introduction
GCTHarmony is an LLM-based method for harmonizing cell type annotations across single-cell
studies. Utilizing OpenAI’s text embedding model, GCTHarmony accurately maps arbitrary cell type annotations to standardized
cell ontology terms and reconciles discrepancies in annotation hierarchies across studies. 
---
## Citation

If you use **GCTHarmony** in your research, please cite:

---

## Contact

* **Authors**: Xingyuan Zhang([xingyuan.zhang@duke.edu](mailto:xingyuan.zhang@duke.edu)), Zhicheng Ji([zhicheng.ji@duke.edu](mailto:zhicheng.ji@duke.edu))
* **GitHub**: [Xingyuan-Zhang](https://github.com/Xingyuan-Zhang)
