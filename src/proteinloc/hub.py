"""Hugging Face Hub integration for proteinloc classifier artifacts.

All network/cache operations related to the HF Hub live here so the rest of
the codebase stays clean and this module is easy to mock in tests.
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default HF repo that hosts the .joblib classifier artifacts.
DEFAULT_HF_REPO: str = "jpuglia/proteinloc"

#: Env-var that lets users point at a fork / mirror.
_ENV_REPO_ID = "PROTEINLOC_HF_REPO_ID"


def resolve_repo_id(repo_id: str | None = None) -> str:
    """Return the repo ID to use, honouring the env-var override."""
    return repo_id or os.getenv(_ENV_REPO_ID) or DEFAULT_HF_REPO


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def resolve_classifier_file(
    filename: str,
    *,
    weights_dir: Path | None = None,
    repo_id: str | None = None,
    revision: str = "main",
) -> Path:
    """Return a local ``Path`` to *filename*, downloading it from the Hub if needed.

    Resolution order:
    1. If *weights_dir* is given **and** ``weights_dir / filename`` exists → return it.
    2. Otherwise download from the Hugging Face Hub using ``hf_hub_download``
       (which transparently caches the file in ``~/.cache/huggingface/hub``).

    Parameters
    ----------
    filename:
        Bare filename of the artifact, e.g. ``"esm_300m_svm.joblib"``.
    weights_dir:
        Optional local directory that overrides HF Hub resolution.
    repo_id:
        HF repo ID.  Defaults to ``jpuglia/proteinloc`` (or the
        ``PROTEINLOC_HF_REPO_ID`` env var).
    revision:
        Git revision / tag on the Hub (default: ``"main"``).

    Returns
    -------
    Path
        Absolute path to the file on the local filesystem.
    """
    # 1. Local override
    if weights_dir is not None:
        local_path = weights_dir / filename
        if local_path.exists():
            return local_path

    # 2. HF Hub download (cached automatically)
    effective_repo = resolve_repo_id(repo_id)
    _console.print(
        f"[dim]Downloading[/dim] [cyan]{filename}[/cyan] "
        f"[dim]from[/dim] [bold]{effective_repo}[/bold][dim]…[/dim]"
    )
    cached = hf_hub_download(
        repo_id=effective_repo,
        filename=filename,
        revision=revision,
    )
    return Path(cached)


def resolve_model_group(
    classifier_file: str,
    encoder_file: str,
    *,
    weights_dir: Path | None = None,
    repo_id: str | None = None,
    revision: str = "main",
) -> tuple[Path, Path]:
    """Resolve both the classifier and label-encoder files for a model group.

    Parameters
    ----------
    classifier_file:
        Filename of the SVM classifier (e.g. ``"esm_300m_svm.joblib"``).
    encoder_file:
        Filename of the label encoder (e.g. ``"esm_300m_le_svm.joblib"``).
    weights_dir:
        Optional local directory override.
    repo_id:
        HF repo ID override.
    revision:
        Git revision on the Hub.

    Returns
    -------
    tuple[Path, Path]
        ``(classifier_path, encoder_path)`` as absolute local paths.
    """
    classifier_path = resolve_classifier_file(
        classifier_file,
        weights_dir=weights_dir,
        repo_id=repo_id,
        revision=revision,
    )
    encoder_path = resolve_classifier_file(
        encoder_file,
        weights_dir=weights_dir,
        repo_id=repo_id,
        revision=revision,
    )
    return classifier_path, encoder_path
