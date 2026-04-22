#Testing Analysis_Notebook

import os
import pytest
from notebook_utils import find_cell

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_W3_LAB = os.path.join(_REPO_ROOT, "Analysis_Notebook.ipynb")


# ===========================================================================
# Tests — Basic SQL query validation
# ===========================================================================

def test_outages_hours_greater_than_100(spark):
    """Verify that only outages with duration greater than 100 hours are inserted."""
    _run_cell(spark, "outage_duration_greater_than_100")
    rows = spark.sql("SELECT * FROM analysis_testing.outages").collect()
    assert len(rows) == 2

# ===========================================================================
# Setup Code
# ===========================================================================

def _run_cell(spark, pattern):
    sql = find_cell(_W3_LAB, pattern)
    assert sql is not None, f"Could not find cell matching: {pattern}"
    return spark.sql(sql)


@pytest.fixture(autouse=True)
def analysis_test_data(spark):
    """Automatically create Analysis schema and tables for all tests.

    This fixture runs before every test in this module, creating the
    schema, tables, and test data that SQL queries will read from.
    """
    spark.sql("CREATE SCHEMA IF NOT EXISTS analysis_testing")

    spark.sql("""
        CREATE OR REPLACE TABLE analysis_testing.cleaned_data (
            event_id INT,
            year_sheet INT,
            outage_duration_hours DOUBLE,
            customers_affected INT,
            demand_loss_mw DOUBLE,
            nerc_region STRING,
            event_type STRING,
            alert_criteria STRING,
            area_affected_raw STRING
        )
    """)

    spark.sql("""
        INSERT INTO analysis_testing.cleaned_data VALUES
            (1, 2019, 10.0, 50, 1000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
            (2, 2020, 20.0, 100, 2000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
            (3, 2021, 30.0, 25.3, 3000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
            (4, 2022, 40.0, 231.2, 4000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
            (5, 2023, 50.0, 3, 5000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas')
    """)
