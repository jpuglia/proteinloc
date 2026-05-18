# proteinloc/utils.py
from pathlib import Path
from enum import Enum
from Bio import SeqIO
from pydantic import BaseModel, Field, field_validator, ValidationError
import re
import typer
from rich.console import Console

console = Console()

# 1. Define the Pydantic schema for data validation
class ProteinSequence(BaseModel):
    id: str = Field(..., description="The FASTA header ID")
    sequence: str = Field(..., description="The raw amino acid sequence")


# 2. Define the parser function that uses Biopython and Pydantic
def validate_and_parse_fasta(file_path: Path) -> list[ProteinSequence]:
    valid_proteins = []
    
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            try:
                protein = ProteinSequence(
                    id=record.id,
                    sequence=str(record.seq)
                )
                valid_proteins.append(protein)
            except ValidationError as e:
                console.print("\n[bold red]❌ Validation Error Detected![/bold red]")
                console.print(f"[yellow]Record ID:[/yellow] {record.id}")
                console.print(f"[yellow]Reason:[/yellow] {e.errors()[0]['msg']}\n")
                raise typer.Exit(code=1)
                
    return valid_proteins