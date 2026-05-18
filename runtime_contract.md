Command:
proteinloc predict

Required Inputs:
- FASTA file path
- model identifier

Optional Inputs:
- output format
- output file
- device selection

Outputs:
- terminal table
- optional JSON/CSV

Execution:
- one-shot
- stateless

Failure Strategy:
- fail fast

Environment:
- local Python runtime

## Architecture

proteinloc/
├── pyproject.toml
├── README.md
├── .gitignore
├── tests/
│
├── src/
│   └── proteinloc/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── commands/
│       │   └── predict.py
│       │
│       ├── validation/
│       │   └── fasta.py
│       │
│       ├── predictor/
│       │   ├── inference.py
│       │   └── model_loader.py
│       │
│       ├── io/
│       │   ├── csv_writer.py
│       │   ├── json_writer.py
│       │   └── table_writer.py
│       │
│       └── models/
│           └── registry.py