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

---

## API Reference

* `GCTHarmony(ontology_path: str, datasets: dict)`: Constructor that loads the ontology and datasets.
* `map_annotations() -> dict`: Returns a dictionary of mapped annotations keyed by dataset name.
* `evaluate_consistency(mapped: dict) -> pandas.DataFrame`: Returns a DataFrame of consistency evaluation metrics.

> For detailed usage and additional parameters, see the [documentation](./docs).

---

## Citation

If you use **iGCTHarmony** in your research, please cite:

```bibtex
@article{zhang2025gctharmony,
  title={GCTHarmony: Harmonizing cell type annotations across single-cell studies},
  author={Zhang, Xingyuan and Ji, Jason},
  journal={Bioinformatics},
  year={2025}
}
```

---

## Contact

* **Email**: [xingyuan.zhang@duke.edu](mailto:xingyuan.zhang@duke.edu)
* **GitHub**: [Xingyuan-Zhang](https://github.com/Xingyuan-Zhang)
