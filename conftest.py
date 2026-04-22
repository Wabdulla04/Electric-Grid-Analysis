import pytest
from pyspark.sql import SparkSession


# ---------------------------
# Spark Session (shared)
# ---------------------------
@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("electric-grid-tests")
        .getOrCreate()
    )
    return spark


# ---------------------------
# Load Cleaned Dataset
# ---------------------------
@pytest.fixture(scope="session")
def cleaned_df(spark):
    df = spark.createDataFrame([
        (1, 2019, 10.0, 50, 1000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (2, 2020, 20.0, 100, 2000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (3, 2021, 30.0, 25.3, 3000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (4, 2022, 40.0, 231.2, 4000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (5, 2023, 50.0, 3, 5000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (6, 2024, 150.0, 500, 6000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (7, 2025, 200.0, 600, 7000000, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
    ], ["event_id", "year_sheet", "outage_duration_hours", "customers_affected", "demand_loss_mw", "nerc_region", "event_type", "alert_criteria", "area_affected_raw"])
    return df


# ---------------------------
# Register as SQL Table
# ---------------------------
@pytest.fixture(scope="session")
def cleaned_table(cleaned_df):
    cleaned_df.createOrReplaceTempView("cleaned_data")
    return "cleaned_data"
