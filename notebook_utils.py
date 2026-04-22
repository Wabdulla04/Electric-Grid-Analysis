"""Utilities for extracting and adapting SQL from .ipynb notebooks."""

import json
import re


def find_cell(notebook_path, tag):
    """Return the SQL source of the cell tagged with `-- @test:tag`.

    Searches through all code cells in the notebook and returns the full
    cell source for the first cell that starts with `-- @test:tag`.
    The tag line itself is stripped from the returned SQL.
    Returns None if no cell matches.
    """
    marker = f"-- @test:{tag}"
    with open(notebook_path) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        # cell source can be a list of lines or a single string
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        if source.startswith(marker):
            # Strip the tag line from the returned SQL
            lines = source.split("\n", 1)
            return lines[1] if len(lines) > 1 else ""
    return None


def get_all_sql_cells(notebook_path):
    """Return a list of all code cell sources from the notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)

    results = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        results.append(source)
    return results


def strip_identity(ddl):
    """Remove GENERATED ALWAYS AS IDENTITY from DDL.

    This makes gold table DDL compatible with open-source Delta Lake /
    local Spark, which doesn't support identity columns. The surrogate
    key columns become plain BIGINT.
    """
    return re.sub(r"\s+GENERATED\s+ALWAYS\s+AS\s+IDENTITY", "", ddl, flags=re.IGNORECASE)
