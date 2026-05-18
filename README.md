# proteinloc

[![PyPI version](https://img.shields.io/pypi/v/proteinloc?label=pypi%20package)](https://pypi.org/project/proteinloc/0.1.0/)
[![Python versions](https://img.shields.io/pypi/pyversions/proteinloc)](https://pypi.org/project/proteinloc/0.1.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/jpuglia/proteinloc/actions/workflows/ci.yml/badge.svg)](https://github.com/jpuglia/proteinloc/actions/workflows/ci.yml)

> Predict protein subcellular localization for **prokaryotic proteins** using state-of-the-art protein language model embeddings — ESM Cambrian and ProstT5.

`proteinloc` classifies sequences into six possible subcellular localizations:
- **Cellwall**
- **Cytoplasmic**
- **CytoplasmicMembrane**
- **Extracellular**
- **OuterMembrane**
- **Periplasmic**

Classifier artifacts (~18–23 MB each) are **not** bundled in the package. They are downloaded from [Hugging Face Hub](https://huggingface.co/jpuglia/proteinloc) on first use and cached locally.

---

## Installation

### Using pip
```bash
pip install proteinloc
```

### Using uv
```bash
uv tool install proteinloc
```

> [!IMPORTANT]
> `proteinloc` requires **Python ≥ 3.10** and **PyTorch ≥ 2.2**.  
> The `esm` package (EvolutionaryScale ESM Cambrian) requires accepting a license agreement on Hugging Face before model weights can be downloaded. Run `huggingface-cli login` and accept the license at [https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12).

---

## Quick start

```bash
# Predict from a FASTA file (classifiers download automatically on first run)
proteinloc predict --fasta sequences.fasta --model esm_300

# Output as JSON
proteinloc predict --fasta sequences.fasta --model esm_600 --output-format json

# Save to a file
proteinloc predict --fasta sequences.fasta --model prost --output-format csv --output results.csv
```

---

## CLI Reference

### `proteinloc predict`
Embed sequences with a protein language model, then classify subcellular localization.

| Option | Description | Default |
|---|---|---|
| `--fasta PATH` | Input FASTA file *(required)* | — |
| <code>--model [esm_300&#124;esm_600&#124;prost]</code> | Embedding model *(required)* | — |
| <code>--output-format [table&#124;json&#124;csv]</code> | Output format | `table` |
| `--output PATH` | Write output to file instead of stdout | — |
| `--device TEXT` | Torch device (`cpu`, `cuda:0`, …) | `auto` |
| `--weights-dir PATH` | Local directory with `.joblib` files (overrides HF Hub) | — |

### `proteinloc models download`
Pre-fetch classifier artifacts for offline use.

```bash
# Download all models
proteinloc models download

# Download a specific model
proteinloc models download --model esm_300
```

### `proteinloc info`
Show authorship, project context, and version information.

### `proteinloc models-list`
List all available embedding models and their classifier details.

### `proteinloc output-formats`
List available output formats and their descriptions.

---

## Authorship & Context

`proteinloc` was developed by **Juan Diego Puglia** as part of a **Degree Thesis in Biotechnology** at **Universidad ORT Uruguay**.

The project aims to leverage the power of Protein Language Models (pLMs) to provide fast and accurate subcellular localization predictions for prokaryotic research.

---

## Developing

To set up a local development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/jpuglia/proteinloc
   cd proteinloc
   ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```

3. Run tests:
   ```bash
   uv run pytest tests/ -v
   ```

---

## License

MIT © Juan Puglia
