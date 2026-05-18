from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from transformers import T5EncoderModel, T5Tokenizer

from proteinloc.tokenizers.base import BaseProteinEmbedder


class ProstT5Embedder(BaseProteinEmbedder):
    """Generate ProstT5 residue-level and per-protein embeddings."""

    def __init__(
        self,
        model_name: str = "Rostlab/ProstT5",
        device: torch.device | str | None = None,
        half_precision: bool | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.half_precision = (
            self.device.type != "cpu" if half_precision is None else half_precision
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name,
            do_lower_case=False,
        )
        self.model = T5EncoderModel.from_pretrained(self.model_name)
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

        if self.half_precision:
            self.model.half()
        else:
            self.model.float()

    @staticmethod
    def prepare_sequence(sequence: str) -> str:
        """Apply ProstT5 token spacing and AA/3Di prefix rules."""
        cleaned_sequence = re.sub(r"[UZOB]", "X", sequence)
        spaced_sequence = " ".join(cleaned_sequence)

        if sequence.isupper():
            return f"<AA2fold> {spaced_sequence}"
        return f"<fold2AA> {spaced_sequence}"

    def prepare_sequences(self, sequences: Sequence[str]) -> list[str]:
        return [self.prepare_sequence(sequence) for sequence in sequences]

    def tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        prepared_sequences = self.prepare_sequences(sequences)
        tokenized = self.tokenizer(
            prepared_sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        return tokenized.to(self.device)

    def embed_sequences(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        """
        Generate one residue-level embedding tensor per sequence.

        Prefix, EOS, and padding tokens are removed from each tensor, so each
        result has shape ``sequence_length x hidden_size``.
        """
        if not sequences:
            return []

        tokenized = self.tokenize(sequences)
        with torch.inference_mode():
            output = self.model(
                tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )

        residue_embeddings: list[torch.Tensor] = []
        for index, sequence in enumerate(sequences):
            sequence_length = len(sequence)
            residue_embeddings.append(
                output.last_hidden_state[index, 1 : sequence_length + 1].detach().cpu()
            )

        return residue_embeddings


if __name__ == "__main__":
    embedder = ProstT5Embedder()
    examples = ["PRTEINO", "strct"]
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
