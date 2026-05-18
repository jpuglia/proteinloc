# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] – 2026-05-17

### Added
- Initial release of the `proteinloc` CLI.
- `predict` command: predict protein subcellular localization from a FASTA file
  using ESM Cambrian (300 M / 600 M) or ProstT5 embeddings.
- `models download` command: pre-fetch classifier artifacts from Hugging Face Hub
  for offline use.
- On-demand classifier download via `huggingface-hub` — no large binary files
  shipped inside the wheel.
- Support for `table`, `json`, and `csv` output formats.
- `--device` flag for explicit Torch device selection (`cpu`, `cuda:0`, …).
- `PROTEINLOC_HF_REPO_ID` environment variable to override the default HF repository.

[Unreleased]: https://github.com/jpuglia/proteinloc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jpuglia/proteinloc/releases/tag/v0.1.0
