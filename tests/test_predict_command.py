from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typer.testing import CliRunner

import proteinloc.commands.predict as predict_module
from proteinloc.cli import app
from proteinloc.classifiers.classifiers import ClassifierLoader
from proteinloc.enums import ModelName

runner = CliRunner()


class FakeESMEmbedder:
    calls: list[dict[str, str | None]] = []

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.calls.append({"model_name": model_name, "device": device})

    def embed_proteins(self, sequences: list[str]) -> list[torch.Tensor]:
        return [torch.tensor([float(index)], dtype=torch.float32) for index, _ in enumerate(sequences)]


class FakeProstT5Embedder:
    calls: list[dict[str, str | None]] = []

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.calls.append({"device": device})

    def embed_proteins(self, sequences: list[str]) -> list[torch.Tensor]:
        return [torch.tensor([float(index)], dtype=torch.float32) for index, _ in enumerate(sequences)]


class FakeClassifierLoader:
    loaded_models: list[ModelName] = []

    def __init__(self, weights_dir: Path | None = None, hf_repo_id: str | None = None) -> None:
        self.weights_dir = weights_dir
        self.hf_repo_id = hf_repo_id

    def load_model_group(self, model: ModelName) -> None:
        self.loaded_models.append(model)

    def predict_probabilities(self, mean_embeddings: list[torch.Tensor]) -> np.ndarray:
        return np.array(
            [
                [0.1, 0.9],
                [0.8, 0.2],
            ][: len(mean_embeddings)],
            dtype=np.float64,
        )

    def decode_predictions(self, probabilities: np.ndarray) -> list[dict[str, float]]:
        return [
            {"Cytoplasmic": float(row[0]), "Extracellular": float(row[1])}
            for row in probabilities
        ]


@pytest.fixture(autouse=True)
def fake_prediction_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeESMEmbedder.calls = []
    FakeProstT5Embedder.calls = []
    FakeClassifierLoader.loaded_models = []

    monkeypatch.setattr(predict_module.esm_module, "ESMEmbedder", FakeESMEmbedder)
    monkeypatch.setattr(predict_module.prost_module, "ProstT5Embedder", FakeProstT5Embedder)
    monkeypatch.setattr(predict_module, "ClassifierLoader", FakeClassifierLoader)


@pytest.fixture
def fasta_file(tmp_path: Path) -> Path:
    path = tmp_path / "proteins.fasta"
    path.write_text(">protein_a\nAAAA\n>protein_b\nMMMM\n", encoding="utf-8")
    return path


def test_predict_outputs_json_and_uses_esm_300(fasta_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta_file),
            "--model",
            "esm_300",
            "--output-format",
            "json",
            "--weights-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeESMEmbedder.calls == [{"model_name": "esmc_300m", "device": "cpu"}]
    assert FakeProstT5Embedder.calls == []
    assert FakeClassifierLoader.loaded_models == [ModelName.esm_300]
    assert '"id": "protein_a"' in result.output
    assert '"prediction": "Extracellular"' in result.output
    assert '"confidence": 0.9' in result.output
    assert '"probabilities": {' in result.output


def test_predict_uses_esm_600_model_name(fasta_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta_file),
            "--model",
            "esm_600",
            "--output-format",
            "json",
            "--weights-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert FakeESMEmbedder.calls == [{"model_name": "esmc_600m", "device": None}]


def test_predict_uses_prost_embedder(fasta_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta_file),
            "--model",
            "prost",
            "--output-format",
            "json",
            "--weights-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert FakeESMEmbedder.calls == []
    assert FakeProstT5Embedder.calls == [{"device": None}]


def test_predict_outputs_csv(fasta_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta_file),
            "--model",
            "prost",
            "--output-format",
            "csv",
            "--weights-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "id,prediction,confidence,Cytoplasmic,Extracellular" in result.output
    assert "protein_a,Extracellular,0.9,0.1,0.9" in result.output
    assert "protein_b,Cytoplasmic,0.8,0.8,0.2" in result.output


def test_predict_outputs_table(fasta_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta_file),
            "--model",
            "prost",
            "--weights-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Protein Localization Predictions" in result.output
    assert "protein_a" in result.output
    assert "Extracellular" in result.output


def test_predict_writes_output_file(fasta_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "predictions.json"
    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta_file),
            "--model",
            "prost",
            "--output-format",
            "json",
            "--output",
            str(output),
            "--weights-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote json predictions" in result.output
    assert '"id": "protein_a"' in output.read_text(encoding="utf-8")


def test_predict_fails_on_empty_fasta(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "predict",
            "--fasta",
            str(fasta),
            "--model",
            "prost",
            "--weights-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "No FASTA records found" in result.output


class ProbabilityEstimator(BaseEstimator):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 2), dtype=np.float64) / 2


def test_classifier_loader_accepts_pipeline_with_predict_proba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLabelEncoder:
        classes_ = np.array(["A", "B"])

    loaded_objects: dict[Path, Any] = {
        tmp_path / "esm_300m_svm.joblib": Pipeline([("classifier", ProbabilityEstimator())]),
        tmp_path / "esm_300m_le_svm.joblib": FakeLabelEncoder(),
    }
    for path in loaded_objects:
        path.touch()

    monkeypatch.setattr(
        "proteinloc.classifiers.classifiers.joblib.load",
        lambda path: loaded_objects[path],
    )
    monkeypatch.setattr(
        "proteinloc.classifiers.classifiers.LabelEncoder",
        FakeLabelEncoder,
    )

    loader = ClassifierLoader(tmp_path)
    loader.load_model_group(ModelName.esm_300)

    assert isinstance(loader.model, Pipeline)
