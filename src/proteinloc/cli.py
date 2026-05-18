from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from proteinloc import __version__
from proteinloc.commands.predict import predict
from proteinloc.commands.models import app as models_app
from proteinloc.enums import ModelName, OutputFormat

console = Console()

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main() -> None:
    """
    [bold cyan]proteinloc[/bold cyan] — Protein subcellular localization predictor.

    Uses protein language model embeddings (ESM Cambrian, ProstT5) combined
    with SVM classifiers to predict where a protein resides inside the cell.

    Run [bold]proteinloc info[/bold] for authorship and project details.
    """


# ── info ──────────────────────────────────────────────────────────────────────

@app.command("info")
def info_command() -> None:
    """Show authorship, project context, and version information."""
    console.print()
    console.print(
        Panel.fit(
            "\n"
            "  [bold cyan]proteinloc[/bold cyan] [dim]v{version}[/dim]\n\n"
            "  [bold]Author[/bold]      Juan Diego Puglia\n"
            "  [bold]Context[/bold]     Degree thesis in Biotechnology\n"
            "  [bold]Institution[/bold] Universidad ORT Uruguay\n\n"
            "  Predicts protein subcellular localization using\n"
            "  protein language model embeddings (ESM Cambrian,\n"
            "  ProstT5) and Support Vector Machine classifiers.\n\n"
            "  [dim]HF Models : https://huggingface.co/jpuglia/proteinloc\n"
            "  Source    : https://github.com/jpuglia/proteinloc[/dim]\n".format(
                version=__version__
            ),
            title="[bold white]About proteinloc[/bold white]",
            border_style="cyan",
        )
    )
    console.print()


# ── models list (top-level shortcut) ──────────────────────────────────────────

@app.command("models-list")
def models_list_command() -> None:
    """List all available embedding models and their classifiers."""
    _print_models_table()


def _print_models_table() -> None:
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

    rows = [
        (ModelName.esm_300.value, "ESM Cambrian (ESMC)", "300 M", "SVM"),
        (ModelName.esm_600.value, "ESM Cambrian (ESMC)", "600 M", "SVM"),
        (ModelName.prost.value,   "ProstT5",             "~1.2 B",  "SVM"),
    ]
    for flag, embedding, params, clf in rows:
        table.add_row(flag, embedding, params, clf)

    console.print()
    console.print(table)
    console.print(
        "  [dim]Classifier artifacts are downloaded automatically from "
        "[cyan]https://huggingface.co/jpuglia/proteinloc[/cyan] on first use.[/dim]"
    )
    console.print()


# ── output-formats ────────────────────────────────────────────────────────────

@app.command("output-formats")
def output_formats_command() -> None:
    """List available output formats for the predict command."""
    table = Table(
        title="Available Output Formats  (--output-format)",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Format", style="bold yellow", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Use with --output", justify="center", style="dim")

    rows = [
        (
            OutputFormat.table.value,
            "Rich formatted table printed to stdout (default)",
            "optional",
        ),
        (
            OutputFormat.json.value,
            "JSON array with id, prediction, confidence, and per-class probabilities",
            "optional",
        ),
        (
            OutputFormat.csv.value,
            "CSV with id, prediction, confidence, and per-class probability columns",
            "optional",
        ),
    ]
    for fmt, desc, out in rows:
        table.add_row(fmt, desc, out)

    console.print()
    console.print(table)
    console.print(
        "  [dim]Example: [bold]proteinloc predict --fasta seqs.fasta "
        "--model esm_300 --output-format csv --output results.csv[/bold][/dim]"
    )
    console.print()


# ── wire up sub-commands ──────────────────────────────────────────────────────

app.command("predict")(predict)
app.add_typer(models_app, name="models")

if __name__ == "__main__":
    app()
