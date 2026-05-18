from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import proteinloc.tokenizers.prostT5 as prost_module
from proteinloc.tokenizers.prostT5 import ProstT5Embedder


class FakeTokenized(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> FakeTokenized:
        self["input_ids"] = self["input_ids"].to(device)
        self["attention_mask"] = self["attention_mask"].to(device)
        return self


class FakeTokenizer:
    requested_model_name: str | None = None
    last_instance: FakeTokenizer | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        do_lower_case: bool,
    ) -> FakeTokenizer:
        assert do_lower_case is False
        cls.requested_model_name = model_name
        cls.last_instance = cls()
        return cls.last_instance

    def __call__(
        self,
        prepared_sequences: list[str],
        *,
        add_special_tokens: bool,
        padding: str,
        return_tensors: str,
    ) -> FakeTokenized:
        assert add_special_tokens is True
        assert padding == "longest"
        assert return_tensors == "pt"

        sequence_lengths = [
            len(prepared_sequence.split()) - 1
            for prepared_sequence in prepared_sequences
        ]
        token_count = max(sequence_lengths, default=0) + 2
        batch_size = len(prepared_sequences)

        return FakeTokenized(
            {
                "input_ids": torch.ones((batch_size, token_count), dtype=torch.long),
                "attention_mask": torch.ones(
                    (batch_size, token_count),
                    dtype=torch.long,
                ),
            }
        )


@dataclass
class FakeModelOutput:
    last_hidden_state: torch.Tensor


class FakeModel:
    requested_model_name: str | None = None
    last_instance: FakeModel | None = None

    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.eval_called = False
        self.float_called = False
        self.half_called = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> FakeModel:
        cls.requested_model_name = model_name
        cls.last_instance = cls()
        return cls.last_instance

    def to(self, device: torch.device) -> None:
        self.device = device

    def eval(self) -> None:
        self.eval_called = True

    def float(self) -> None:
        self.float_called = True

    def half(self) -> None:
        self.half_called = True

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> FakeModelOutput:
        assert input_ids.shape == attention_mask.shape

        batch_size, token_count = input_ids.shape
        hidden_size = 4
        embeddings = torch.arange(
            batch_size * token_count * hidden_size,
            dtype=torch.float32,
        ).reshape(batch_size, token_count, hidden_size)
        return FakeModelOutput(last_hidden_state=embeddings)


@pytest.fixture
def fake_prostt5_components(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTokenizer.requested_model_name = None
    FakeTokenizer.last_instance = None
    FakeModel.requested_model_name = None
    FakeModel.last_instance = None

    monkeypatch.setattr(prost_module, "T5Tokenizer", FakeTokenizer)
    monkeypatch.setattr(prost_module, "T5EncoderModel", FakeModel)


def test_to_sklearn_features_returns_mean_pooled_numpy_matrix(
    fake_prostt5_components: None,
) -> None:
    embedder = ProstT5Embedder(model_name="fake-prostt5", device="cpu")

    features = embedder.to_sklearn_features(["ACD", "M"])

    assert features.shape == (2, 4)
    assert features.dtype == "float32"
    assert features.tolist() == [
        [8.0, 9.0, 10.0, 11.0],
        [24.0, 25.0, 26.0, 27.0],
    ]


def test_to_sklearn_features_returns_empty_matrix(
    fake_prostt5_components: None,
) -> None:
    embedder = ProstT5Embedder(model_name="fake-prostt5", device="cpu")

    features = embedder.to_sklearn_features([])

    assert features.shape == (0, 0)
    assert features.dtype == "float32"
