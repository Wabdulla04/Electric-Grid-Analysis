import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType


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

schema = StructType([
    StructField("event_id", LongType(),   True),
    StructField("year_sheet", LongType(),   True),
    StructField("outage_duration_hours", DoubleType(), True),
    StructField("customers_affected", DoubleType(),   True),
    StructField("demand_loss_mw", DoubleType(),   True),
    StructField("nerc_region", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("alert_criteria", StringType(), True),
    StructField("area_affected_raw", StringType(), True),
])

@pytest.fixture(scope="session")
def cleaned_df(spark):
    df = spark.createDataFrame([
        (1, 2019, 10.0,   50.0,  1000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (2, 2020, 20.0,  100.0,  2000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (3, 2021, 30.0,   25.3,  3000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (4, 2022, 40.0,  231.2,  4000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (5, 2023, 50.0,    3.0,  5000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (6, 2024, 150.0, 500.0,  6000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
        (7, 2025, 200.0, 600.0,  7000000.0, 'ERCOT', 'Outage', 'Customers Affected', 'Texas'),
    ], schema=schema)
    return df


# ---------------------------
# Register as SQL Table
# ---------------------------
@pytest.fixture(scope="session")
def cleaned_table(cleaned_df):
    cleaned_df.createOrReplaceTempView("cleaned_data")
    return "cleaned_data"
