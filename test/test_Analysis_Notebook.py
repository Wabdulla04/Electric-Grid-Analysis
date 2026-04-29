# Testing Analysis_Notebook
import os
import pytest
from notebook_utils import find_cell

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_W3_LAB = os.path.join(_REPO_ROOT, "src/Analysis_Notebook.ipynb")

# ===========================================================================
# Tests — Basic SQL query validation
# ===========================================================================
def test_outages_hours_greater_than_100(spark, cleaned_table):
    df = spark.sql("""
        SELECT * FROM cleaned_data
        WHERE outage_duration_hours > 100
    """)
    rows = df.collect()
    assert len(rows) == 2

# ===========================================================================
# Notebook cell runner
# ===========================================================================
def _run_cell(spark, pattern):
    sql = find_cell(_W3_LAB, pattern)
    assert sql is not None, f"Could not find cell matching: {pattern}"
    return spark.sql(sql)
