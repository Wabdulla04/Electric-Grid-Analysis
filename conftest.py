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
# Volume Path
# ---------------------------
@pytest.fixture(scope="session")
def volume_path():
    return "/Volumes/electric_grid/data_schema/volume_set"


# ---------------------------
# Load Cleaned Dataset
# ---------------------------
@pytest.fixture(scope="session")
def cleaned_df(spark, volume_path):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{volume_path}/doe_events_db_ready.csv")
    )
    return df


# ---------------------------
# Register as SQL Table
# ---------------------------
@pytest.fixture(scope="session")
def cleaned_table(spark, cleaned_df):
    table_name = "cleaned_data"
    cleaned_df.createOrReplaceTempView(table_name)
    return table_name


# ---------------------------
# Optional: Population Dataset (if needed later)
# ---------------------------
@pytest.fixture(scope="session")
def population_df(spark, volume_path):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{volume_path}/population.csv")
    )
