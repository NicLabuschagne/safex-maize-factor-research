"""
Build a shareable, code-free HTML report from a notebook.

Run from the terminal after the notebook has been executed:

    python export_report.py                        # seasonality.ipynb
    python export_report.py notebooks/other.ipynb  # anything else

Produces notebooks/<name>_report_dark.html and _light.html — markdown and
outputs only, no code, images embedded so the file travels on its own.

The report starts at the notebook's title: the first markdown cell beginning
with a single '#'. Everything before it is setup and plumbing whose outputs
read as noise in a document sent to someone else. Anchoring on the title rather
than a cell number means inserting or deleting setup cells cannot silently
lop the first section off the report.

There is no direct PDF path on this machine (no LaTeX, no playwright).
Open the HTML in a browser and print to PDF — it paginates cleanly.
"""

from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat

ROOT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "seasonality.ipynb"
THEMES = ("dark", "light")


def find_title(nb) -> int:
    """Index of the first markdown cell that opens with a top-level heading."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        first = cell.source.strip().splitlines()[0].strip() if cell.source.strip() else ""
        if first.startswith("# "):
            return i
    return 0


def stage(notebook: Path, target: Path) -> tuple[int, str]:
    """Write a copy of `notebook` holding the report section only."""
    nb = nbformat.read(notebook, as_version=4)
    start = find_title(nb)
    title = nb.cells[start].source.strip().splitlines()[0].lstrip("# ").strip()

    nb.cells = [
        cell
        for cell in nb.cells[start:]
        if cell.source.strip() and (cell.cell_type == "markdown" or cell.get("outputs"))
    ]
    nbformat.validate(nb)
    nbformat.write(nb, target)
    return len(nb.cells), title


def convert(staged: Path, out: Path, theme: str) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--no-input",
            "--embed-images",
            "--theme", theme,
            "--output", str(out),
            str(staged),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    notebook = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    if not notebook.exists():
        raise SystemExit(f"notebook not found: {notebook}")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / notebook.name
        n_cells, title = stage(notebook, staged)
        print(f'{notebook.name}: {n_cells} cells, starting at "{title}"')

        for theme in THEMES:
            out = notebook.parent / f"{notebook.stem}_report_{theme}.html"
            convert(staged, out, theme)
            print(f"  {out.name}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()