import os
import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from proteinloc.classifiers.classifiers import ClassifierLoader
from proteinloc.commands.models import DEFAULT_HF_REPO
from huggingface_hub import HfApi, create_repo

def stage_bundle(models_dir: Path, staging_dir: Path, repo_id: str | None = None, token: str | None = None) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    active_files = []
    for pair in ClassifierLoader.CLASSIFIERS.values():
        active_files.extend(pair)

    print(f"Staging {len(active_files)} artifacts to {staging_dir}...")
    
    for filename in active_files:
        src = models_dir / filename
        if not src.exists():
            print(f"Warning: {filename} not found in {models_dir}")
            continue
        shutil.copy2(src, staging_dir / filename)

    manifest = {
        "version": "0.1.0",
        "active_artifacts": list(ClassifierLoader.CLASSIFIERS.keys()),
        "files": active_files
    }
    
    with open(staging_dir / "artifact_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    readme_content = f"""---
license: mit
library_name: proteinloc
tags:
- biology
- protein-localization
---
# ProteinLoc Model Artifacts

This repository contains the trained SVM classifiers and label encoders for the `proteinloc` CLI.

## Artifacts
{chr(10).join([f"- {f}" for f in active_files])}

## Usage
These files are automatically managed by the `proteinloc` CLI.

```bash
pip install proteinloc
proteinloc predict --fasta sequences.fasta --model esm_300
```
"""
    with open(staging_dir / "README.md", "w") as f:
        f.write(readme_content)

    if repo_id:
        print(f"Uploading artifacts to {repo_id}...")
        api = HfApi(token=token)
        
        # Ensure repository exists
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

app = typer.Typer(help="Stage and upload model artifacts to Hugging Face.")

@app.command()
def run(
    models_dir: Path = typer.Option(Path("models"), help="Directory containing .joblib files."),
    staging_dir: Path = typer.Option(Path("hf_staging"), help="Directory to prepare for upload."),
    repo_id: Optional[str] = typer.Option(
        None,
        help=f"Hugging Face repository ID. Defaults to {DEFAULT_HF_REPO} if triggered.",
    ),
    upload: bool = typer.Option(False, "--upload", help="Whether to trigger the upload after staging."),
    token: Optional[str] = typer.Option(
        None,
        help="Hugging Face write token. If omitted, uses local cache or HF_TOKEN env var.",
    ),
) -> None:
    """Gather artifacts and optionally push to the Hub."""
    hf_token = token or os.getenv("HF_TOKEN")
    target_repo = repo_id or (DEFAULT_HF_REPO if upload else None)
    stage_bundle(models_dir, staging_dir, target_repo if upload else None, hf_token)

if __name__ == "__main__":
    app()