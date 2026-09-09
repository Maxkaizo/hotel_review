"""Ejecutar la libreta y conservar salidas incluso si una celda falla.

uv run python scripts/execute_notebook.py --prepare-only
uv run python scripts/execute_notebook.py
"""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="clasificador_resenas_hoteles.ipynb",
                        help="Nombre de la libreta en la raíz del proyecto")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validar hasta las secuencias, sin descargar Word2Vec")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for variable, subdirectory in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython")]:
        directory = root / ".cache" / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(directory))
    path = (root / args.notebook).resolve()
    if path.parent != root or path.suffix != ".ipynb":
        parser.error("La libreta debe ser un archivo .ipynb de la raíz del proyecto")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}})
    try:
        with client.setup_kernel():
            for index, cell in enumerate(notebook.cells):
                if args.prepare_only and cell.cell_type == "markdown" and cell.source.startswith("## 4."):
                    break
                if cell.cell_type == "code":
                    print(f"Ejecutando celda {index}: {notebook.cells[index - 1].source.splitlines()[0]}", flush=True)
                    client.execute_cell(cell, index)
                    nbformat.write(notebook, path)
    finally:
        nbformat.write(notebook, path)
    print("Ejecución solicitada completada.", flush=True)


if __name__ == "__main__":
    main()
