from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from proteinloc.tokenizers.base import BaseProteinEmbedder


class ESMEmbedder(BaseProteinEmbedder):
    """Generate ESM Cambrian residue-level and per-protein embeddings."""

    def __init__(
        self,
        model_name: str = "esmc_300m",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)

        esm_components = self._load_esm_components()
        self._protein_cls = esm_components["protein_cls"]
        self._protein_error_cls = esm_components["protein_error_cls"]
        self._logits_config = esm_components["logits_config_cls"](
            sequence=True,
            return_embeddings=True,
        )

        try:
            self.client = esm_components["client_cls"].from_pretrained(self.model_name)
        except Exception as e:
            if type(e).__name__ == "GatedRepoError":
                raise RuntimeError(
                    f"Access to the ESM model ({self.model_name}) is restricted.\n"
                    "Please accept the license at https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1\n"
                    "and authenticate by running `huggingface-cli login` in your terminal."
                ) from e
            raise
        
        self.client = self.client.to(str(self.device))
        if hasattr(self.client, "eval"):
            self.client.eval()

    @staticmethod
    def _load_esm_components() -> dict[str, Any]:
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
        except ImportError as error:
            message = (
                "The `esm` package is required for ESM embeddings. "
                "Install it with `uv add esm` or add `esm` to pyproject.toml."
            )
            raise ImportError(message) from error

        return {
            "client_cls": ESMC,
            "protein_cls": ESMProtein,
            "protein_error_cls": ESMProteinError,
            "logits_config_cls": LogitsConfig,
        }

    def embed_sequences(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        """
        Generate one residue-level embedding tensor per sequence.

        Special tokens are removed from each tensor, so each result has shape
        ``sequence_length x hidden_size``.
        """
        if not sequences:
            return []

        return [self._embed_sequence(sequence) for sequence in sequences]

    def _embed_sequence(self, sequence: str) -> torch.Tensor:
        protein = self._protein_cls(sequence=sequence)
        protein_tensor = self.client.encode(protein)

        if isinstance(protein_tensor, self._protein_error_cls):
            raise ValueError(protein_tensor)

        with torch.inference_mode():
            output = self.client.logits(protein_tensor, self._logits_config)

        embeddings = output.embeddings
        if embeddings is None:
            raise ValueError("ESM returned no embeddings.")

        return self._remove_special_tokens(embeddings, len(sequence)).detach().cpu()

    @staticmethod
    def _remove_special_tokens(
        embeddings: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        if embeddings.dim() == 3:
            return embeddings[0, 1 : sequence_length + 1]
        return embeddings[1 : sequence_length + 1]


if __name__ == "__main__":
    embedder = ESMEmbedder()
    examples = ["AAAAA"]
    embeddings = embedder.embed_sequences(examples)
    protein_embeddings = embedder.embed_proteins(examples)

    for sequence, embedding, protein_embedding in zip(
        examples,
        embeddings,
        protein_embeddings,
        strict=True,
    ):
        print(
            sequence,
            tuple(embedding.shape),
            tuple(protein_embedding.shape),
        )
