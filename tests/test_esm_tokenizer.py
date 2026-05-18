from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from proteinloc.tokenizers.esm import ESMEmbedder


@dataclass
class FakeProtein:
    sequence: str


class FakeProteinError:
    pass


@dataclass
class FakeProteinTensor:
    sequence: str


@dataclass
class FakeLogitsOutput:
    embeddings: torch.Tensor | None


class FakeLogitsConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeClient:
    last_instance: FakeClient | None = None
    requested_model_name: str | None = None

    def __init__(self) -> None:
        self.device: str | None = None
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> FakeClient:
        cls.requested_model_name = model_name
        cls.last_instance = cls()
        return cls.last_instance

    def to(self, device: str) -> FakeClient:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def encode(self, protein: FakeProtein) -> FakeProteinTensor | FakeProteinError:
        if protein.sequence == "BAD":
            return FakeProteinError()
        return FakeProteinTensor(sequence=protein.sequence)

    def logits(
        self,
        protein_tensor: FakeProteinTensor,
        logits_config: FakeLogitsConfig,
    ) -> FakeLogitsOutput:
        assert logits_config.kwargs == {"sequence": True, "return_embeddings": True}
        if protein_tensor.sequence == "NONE":
            return FakeLogitsOutput(embeddings=None)

        token_count = len(protein_tensor.sequence) + 2
        embeddings = torch.arange(token_count * 4, dtype=torch.float32).reshape(
            1,
            token_count,
            4,
        )
        return FakeLogitsOutput(embeddings=embeddings)


@pytest.fixture
def fake_esm_components(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.last_instance = None
    FakeClient.requested_model_name = None

    monkeypatch.setattr(
        ESMEmbedder,
        "_load_esm_components",
        staticmethod(
            lambda: {
                "client_cls": FakeClient,
                "protein_cls": FakeProtein,
                "protein_error_cls": FakeProteinError,
                "logits_config_cls": FakeLogitsConfig,
            }
        ),
    )


def test_init_loads_model_on_device_and_sets_eval(fake_esm_components: None) -> None:
    embedder = ESMEmbedder(model_name="fake-esm", device="cpu")

    assert embedder.model_name == "fake-esm"
    assert FakeClient.requested_model_name == "fake-esm"
    assert FakeClient.last_instance is not None
    assert FakeClient.last_instance.device == "cpu"
    assert FakeClient.last_instance.eval_called is True


def test_embed_sequences_removes_special_tokens(
    fake_esm_components: None,
) -> None:
    embedder = ESMEmbedder(device="cpu")

    embeddings = embedder.embed_sequences(["ACD", "M"])

    assert [tuple(embedding.shape) for embedding in embeddings] == [(3, 4), (1, 4)]
    assert torch.equal(
        embeddings[0],
        torch.tensor(
            [
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]
        ),
    )
    assert embeddings[0].device.type == "cpu"
    assert embeddings[0].requires_grad is False


def test_embed_sequences_returns_empty_list(fake_esm_components: None) -> None:
    embedder = ESMEmbedder(device="cpu")

    assert embedder.embed_sequences([]) == []


def test_embed_sequence_raises_for_esm_protein_error(
    fake_esm_components: None,
) -> None:
    embedder = ESMEmbedder(device="cpu")

    with pytest.raises(ValueError, match="FakeProteinError"):
        embedder.embed_sequence("BAD")


def test_embed_sequence_raises_when_esm_returns_no_embeddings(
    fake_esm_components: None,
) -> None:
    embedder = ESMEmbedder(device="cpu")

    with pytest.raises(ValueError, match="ESM returned no embeddings"):
        embedder.embed_sequence("NONE")


def test_remove_special_tokens_accepts_two_dimensional_embeddings() -> None:
    embeddings = torch.arange(5 * 3).reshape(5, 3)

    result = ESMEmbedder._remove_special_tokens(embeddings, sequence_length=2)

    assert torch.equal(result, embeddings[1:3])


def test_embed_proteins_mean_pools_residue_embeddings(
    fake_esm_components: None,
) -> None:
    embedder = ESMEmbedder(device="cpu")

    protein_embedding = embedder.embed_protein("ACD")

    assert torch.equal(protein_embedding, torch.tensor([8.0, 9.0, 10.0, 11.0]))


def test_to_sklearn_features_returns_mean_pooled_numpy_matrix(
    fake_esm_components: None,
) -> None:
    embedder = ESMEmbedder(device="cpu")

    features = embedder.to_sklearn_features(["ACD", "M"])

    assert features.shape == (2, 4)
    assert features.dtype == "float32"
    assert features.tolist() == [
        [8.0, 9.0, 10.0, 11.0],
        [4.0, 5.0, 6.0, 7.0],
    ]


def test_to_sklearn_features_returns_empty_matrix(
    fake_esm_components: None,
) -> None:
    embedder = ESMEmbedder(device="cpu")

    features = embedder.to_sklearn_features([])

    assert features.shape == (0, 0)
    assert features.dtype == "float32"
