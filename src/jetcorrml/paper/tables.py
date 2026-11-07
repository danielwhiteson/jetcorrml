"""
The paper's tables as data, then as text.

Each table program computes its rows of numbers, then renders the text from those
rows.  ``write`` puts both under output/paper/tables/: <name>.csv holds the rows
(the numbers at full precision; the run compares these), and <name>.tex (or .txt
for numbers quoted in the text) holds what the paper shows.
"""

import csv
from pathlib import Path

from jetcorrml import paths


def write(name, rows, text, ext="tex"):
    """rows: a list of dicts, one per line of the table; text: the rendered lines."""
    out = Path(paths.paper_table(name, "csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in keys})
    body = "\n".join(text) + "\n"
    Path(paths.paper_table(name, ext)).write_text(body)
    print(body, end="")


def read(path):
    """The rows of a table's csv, every cell as written (text)."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
