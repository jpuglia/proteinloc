from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch


class BaseProteinEmbedder(ABC):
    """Shared interface and state for protein embedding models."""

    def __init__(
        self,
        model_name: str,
        device: torch.device | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_sequences(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        """Generate one residue-level embedding tensor per sequence."""

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        return self.embed_sequences([sequence])[0]

    def embed_proteins(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        """Generate one mean-pooled embedding tensor per sequence."""
        return [embedding.mean(dim=0) for embedding in self.embed_sequences(sequences)]

    def embed_protein(self, sequence: str) -> torch.Tensor:
        return self.embed_proteins([sequence])[0]

    def to_sklearn_features(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        """
        Generate a 2D NumPy feature matrix ready for scikit-learn classifiers.

        Each sequence is embedded, mean-pooled to one protein-level vector, and
        stacked into an array with shape ``n_sequences x hidden_size``.
        """
        protein_embeddings = self.embed_proteins(sequences)
        if not protein_embeddings:
            return np.empty((0, 0), dtype=np.float32)

        return torch.stack(protein_embeddings).numpy().astype(np.float32, copy=False)
