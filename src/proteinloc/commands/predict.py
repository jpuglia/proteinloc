from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TypedDict

import typer
from Bio import SeqIO
from rich.console import Console
from rich.table import Table

from . import __version__
from proteinloc.classifiers.classifiers import ClassifierLoader
from proteinloc.enums import ModelName, OutputFormat
from proteinloc.tokenizers import esm as esm_module
from proteinloc.tokenizers import prostT5 as prost_module

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
)
stdout_console = Console()
error_console = Console(stderr=True)

ESM_MODEL_NAMES: dict[ModelName, str] = {
    ModelName.esm_300: "esmc_300m",
    ModelName.esm_600: "esmc_600m",
}


@dataclass(frozen=True)
class FastaRecord:
    id: str
    sequence: str


class PredictionResult(TypedDict):
    id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]


def _parse_fasta(fasta: Path) -> list[FastaRecord]:
    records = [
        FastaRecord(id=record.id, sequence=str(record.seq))
        for record in SeqIO.parse(fasta, "fasta")
    ]

    if not records:
        raise typer.BadParameter(f"No FASTA records found in {fasta}.")

    empty_records = [record.id for record in records if not record.sequence]
    if empty_records:
        raise typer.BadParameter(
            f"FASTA records must contain sequences. Empty records: {', '.join(empty_records)}"
        )

    return records


def _build_embedder(model: ModelName, device: str | None) -> Any:
    if model in ESM_MODEL_NAMES:
        return esm_module.ESMEmbedder(
            model_name=ESM_MODEL_NAMES[model],
            device=device,
        )

    return prost_module.ProstT5Embedder(device=device)


def _build_results(
    records: list[FastaRecord],
    probability_maps: list[dict[str, float]],
) -> list[PredictionResult]:
    results: list[PredictionResult] = []

    for record, probabilities in zip(records, probability_maps, strict=True):
        prediction, confidence = max(probabilities.items(), key=lambda item: item[1])
        results.append(
            {
                "id": record.id,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )

    return results


def _class_names(results: list[PredictionResult]) -> list[str]:
    return list(results[0]["probabilities"])


def _results_to_csv(results: list[PredictionResult]) -> str:
    if not results:
        return ""

    class_names = _class_names(results)
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["id", "prediction", "confidence", *class_names],
    )
    writer.writeheader()

    for result in results:
        probabilities = result["probabilities"]

        writer.writerow(
            {
                "id": result["id"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                **probabilities,
            }
        )

    return buffer.getvalue()


def _results_to_json(results: list[PredictionResult]) -> str:
    return json.dumps(results, indent=2)


def _render_table(results: list[PredictionResult]) -> Table:
    table = Table(title="Protein Localization Predictions")
    table.add_column("Protein ID", style="cyan")
    table.add_column("Prediction", style="green")
    table.add_column("Confidence", justify="right")

    class_names = _class_names(results)
    for class_name in class_names:
        table.add_column(class_name, justify="right")

    for result in results:
        probabilities = result["probabilities"]

        table.add_row(
            str(result["id"]),
            str(result["prediction"]),
            f"{float(result['confidence']):.4f}",
            *[f"{probabilities[class_name]:.4f}" for class_name in class_names],
        )

    return table


def _write_or_print(
    content: str | Table,
    output: Path | None,
    output_format: OutputFormat,
) -> None:
    if output is None:
        if isinstance(content, Table):
            stdout_console.print(content)
        else:
            typer.echo(content)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, Table):
        file_console = Console(record=True, width=160)
        file_console.print(content)
        output.write_text(file_console.export_text(), encoding="utf-8")
    else:
        output.write_text(content, encoding="utf-8")

    stdout_console.print(f"Wrote {output_format.value} predictions to {output}")


@app.command()
def predict(
    fasta: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to FASTA file.",
    ),
    model: ModelName = typer.Option(
        ...,
        help="Model to use for inference.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.table,
        help="Prediction output format.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Optional output file path.",
    ),
    device: Optional[str] = typer.Option(
        None,
        help="Torch device to use for embeddings, e.g. 'cpu' or 'cuda:0'.",
    ),
    weights_dir: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help=(
            "Local directory containing .joblib classifier files. "
            "If omitted, classifiers are fetched from Hugging Face Hub "
            "on first use and cached in ~/.cache/huggingface/hub."
        ),
    ),
    hf_repo_id: Optional[str] = typer.Option(
        None,
        envvar="PROTEINLOC_HF_REPO_ID",
        help="Hugging Face repo ID for classifier artifacts.",
        hidden=True,
    ),
) -> None:
    """Predict protein localization from a FASTA file."""
    try:
        records = _parse_fasta(fasta)
        sequences = [record.sequence for record in records]

        embedder = _build_embedder(model, device)
        mean_embeddings = embedder.embed_proteins(sequences)

        classifier = ClassifierLoader(
            weights_dir=weights_dir,
            hf_repo_id=hf_repo_id,
        )
        classifier.load_model_group(model)
        probabilities = classifier.predict_probabilities(mean_embeddings)
        probability_maps = classifier.decode_predictions(probabilities)

        results = _build_results(records, probability_maps)

        if output_format is OutputFormat.json:
            content: str | Table = _results_to_json(results)
        elif output_format is OutputFormat.csv:
            content = _results_to_csv(results)
        else:
            content = _render_table(results)

        _write_or_print(content, output, output_format)
    except (
        typer.BadParameter,
        FileNotFoundError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        error_console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
