from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast

import joblib
import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import LabelEncoder

from proteinloc import hub
from proteinloc.enums import ModelName


class ProbabilityClassifier(Protocol):
    """A scikit-learn compatible classifier that supports class probabilities."""

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float64]: ...


class ClassifierLoader:
    """Handles loading pairs of trained scikit-learn models and label encoders.

    If *weights_dir* is ``None`` (the default), classifier files are resolved
    from the Hugging Face Hub cache, downloading them on first use.  Pass an
    explicit directory to use local files instead (useful for offline or
    custom-trained models).
    """

    #: Maps each ModelName to (classifier_filename, label_encoder_filename).
    CLASSIFIERS: Dict[ModelName, Tuple[str, str]] = {
        ModelName.esm_300: ("esm_300m_svm.joblib", "esm_300m_le_svm.joblib"),
        ModelName.esm_600: ("esm_600m_svm.joblib", "esm_600m_le_svm.joblib"),
        ModelName.prost: ("Prost T5_svm.joblib", "Prost T5_le_svm.joblib"),
    }

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        *,
        hf_repo_id: Optional[str] = None,
        hf_revision: str = "main",
    ) -> None:
        self.weights_dir = weights_dir
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.model: ProbabilityClassifier | None = None
        self.label_encoder: LabelEncoder | None = None

    def load_model_group(self, model_key: ModelName) -> None:
        """Load the classifier and label encoder for *model_key*.

        Files are resolved in order:
        1. ``weights_dir`` (if provided and the file exists there).
        2. Hugging Face Hub cache (download on first use).
        """
        filenames = self.CLASSIFIERS.get(model_key)
        if not filenames:
            raise KeyError(f"Model key '{model_key}' is not registered in CLASSIFIERS.")

        classifier_file, encoder_file = filenames

        model_path, encoder_path = hub.resolve_model_group(
            classifier_file,
            encoder_file,
            weights_dir=self.weights_dir,
            repo_id=self.hf_repo_id,
            revision=self.hf_revision,
        )

        loaded_model = joblib.load(model_path)
        loaded_encoder = joblib.load(encoder_path)

        if not hasattr(loaded_model, "predict_proba"):
            raise TypeError("Loaded classifier does not support predict_proba.")
        if not isinstance(loaded_encoder, LabelEncoder):
            raise TypeError("Loaded label encoder is not a sklearn LabelEncoder.")

        self.model = cast(ProbabilityClassifier, loaded_model)
        self.label_encoder = loaded_encoder

    def predict_probabilities(
        self,
        mean_embeddings: List[torch.Tensor],
    ) -> npt.NDArray[np.float64]:
        """Generate downstream class probabilities from pooled embeddings."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded yet. Call load_model_group first.")

        features: npt.NDArray[np.float32] = torch.stack(mean_embeddings).numpy()
        return self.model.predict_proba(features)

    def decode_predictions(
        self,
        probabilities: npt.NDArray[np.float64],
    ) -> List[Dict[str, float]]:
        """Map raw numerical probabilities to location name dictionaries."""
        if self.label_encoder is None:
            raise RuntimeError("Label encoder has not been loaded yet.")

        location_names: npt.NDArray[Any] = self.label_encoder.classes_
        all_results: List[Dict[str, float]] = []

        for prob_array in probabilities:
            protein_mapping = {
                str(location_names[i]): float(prob_array[i])
                for i in range(len(location_names))
            }
            all_results.append(protein_mapping)

        return all_results
