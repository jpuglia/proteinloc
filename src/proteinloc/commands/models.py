from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from proteinloc import hub
from proteinloc.classifiers.classifiers import ClassifierLoader
from proteinloc.enums import ModelName

app = typer.Typer(
    help="Manage local classifier artifacts.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)

_MODEL_DETAILS: dict[str, tuple[str, str, str]] = {
    ModelName.esm_300.value: ("ESM Cambrian (ESMC)", "300 M",  "SVM"),
    ModelName.esm_600.value: ("ESM Cambrian (ESMC)", "600 M",  "SVM"),
    ModelName.prost.value:   ("ProstT5",             "~1.2 B", "SVM"),
}


@app.command("list")
def list_models() -> None:
    """List all available embedding models and their classifiers."""
    table = Table(
        title="Available Models",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Flag (--model)", style="bold yellow", no_wrap=True)
    table.add_column("Embedding Model", style="white")
    table.add_column("Parameters", justify="right", style="dim")
    table.add_column("Classifier", style="green")

    for flag, (embedding, params, clf) in _MODEL_DETAILS.items():
        table.add_row(flag, embedding, params, clf)

    console.print()
    console.print(table)
    console.print(
        "  [dim]Artifacts are fetched from "
        "[cyan]https://huggingface.co/jpuglia/proteinloc[/cyan] on first use.[/dim]"
    )
    console.print()


@app.command()
def download(
    model: Optional[ModelName] = typer.Option(
        None,
        help=(
            "Specific model to download. "
            "If omitted, downloads all active classifier artifacts."
        ),
    ),
    hf_repo_id: Optional[str] = typer.Option(
        None,
        envvar="PROTEINLOC_HF_REPO_ID",
        help="Hugging Face repo ID (default: jpuglia/proteinloc).",
    ),
    revision: str = typer.Option(
        "main",
        help="Git revision / tag on the Hub.",
    ),
) -> None:
    """Pre-fetch classifier artifacts from Hugging Face Hub for offline use."""
    repo_id = hf_repo_id or os.getenv("PROTEINLOC_HF_REPO_ID") or hub.DEFAULT_HF_REPO
    targets: dict[ModelName, tuple[str, str]]

    if model:
        targets = {model: ClassifierLoader.CLASSIFIERS[model]}
    else:
        targets = dict(ClassifierLoader.CLASSIFIERS)

    console.print(
        f"[bold]Downloading classifiers from[/bold] [cyan]{repo_id}[/cyan] "
        f"([dim]revision: {revision}[/dim])"
    )

    try:
        for model_key, (clf_file, enc_file) in targets.items():
            console.print(f"  [dim]→[/dim] [yellow]{model_key.value}[/yellow]")
            clf_path, enc_path = hub.resolve_model_group(
                clf_file,
                enc_file,
                repo_id=repo_id,
                revision=revision,
            )
            console.print(f"    classifier : [green]{clf_path}[/green]")
            console.print(f"    encoder    : [green]{enc_path}[/green]")

        console.print("\n[bold green]✓ Done.[/bold green]")
    except Exception as exc:
        error_console.print(f"[bold red]Error downloading artifacts:[/bold red] {exc}")
        raise typer.Exit(1) from exc