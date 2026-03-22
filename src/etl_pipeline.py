"""
House Sale Data ETL Pipeline
============================
Implement the three functions below to complete the ETL pipeline.

Steps:
  1. EXTRACT  – load the CSV into a PySpark DataFrame
  2. TRANSFORM – split the data by neighborhood and save each as a separate CSV
  3. LOAD      – insert each neighborhood DataFrame into its own PostgreSQL table
"""
from __future__ import annotations

import csv  # noqa: F401
import os  # noqa: F401
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401
from pyspark.sql import DataFrame, SparkSession  # noqa: F401
from pyspark.sql import functions as F  # noqa: F401

# ── Predefined constants (do not modify) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

NEIGHBORHOODS = [
    "Downtown", "Green Valley", "Hillcrest", "Lakeside", "Maple Heights",
    "Oakwood", "Old Town", "Riverside", "Suburban Park", "University District",
]

OUTPUT_DIR   = ROOT / "output" / "by_neighborhood"
OUTPUT_FILES = {hood: OUTPUT_DIR / f"{hood.replace(' ', '_').lower()}.csv" for hood in NEIGHBORHOODS}

PG_TABLES = {hood: f"public.{hood.replace(' ', '_').lower()}" for hood in NEIGHBORHOODS}

PG_COLUMN_SCHEMA = (
    "house_id TEXT, neighborhood TEXT, price INTEGER, square_feet INTEGER, "
    "num_bedrooms INTEGER, num_bathrooms INTEGER, house_age INTEGER, "
    "garage_spaces INTEGER, lot_size_acres NUMERIC(6,2), has_pool BOOLEAN, "
    "recently_renovated BOOLEAN, energy_rating TEXT, location_score INTEGER, "
    "school_rating INTEGER, crime_rate INTEGER, "
    "distance_downtown_miles NUMERIC(6,2), sale_date DATE, days_on_market INTEGER"
)


def extract(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load the CSV dataset into a PySpark DataFrame with correct data types."""
    df = (
        spark.read.option("header", True).csv(csv_path)
        # Cast numeric columns
        .withColumn("price", F.col("price").cast("int"))
        .withColumn("square_feet", F.col("square_feet").cast("int"))
        .withColumn("num_bedrooms", F.col("num_bedrooms").cast("int"))
        .withColumn("num_bathrooms", F.col("num_bathrooms").cast("int"))
        .withColumn("house_age", F.col("house_age").cast("int"))
        .withColumn("garage_spaces", F.col("garage_spaces").cast("int"))
        .withColumn("lot_size_acres", F.col("lot_size_acres").cast("decimal(6,2)"))
        .withColumn("location_score", F.col("location_score").cast("int"))
        .withColumn("school_rating", F.col("school_rating").cast("int"))
        .withColumn("crime_rate", F.col("crime_rate").cast("int"))
        .withColumn("distance_downtown_miles", F.col("distance_downtown_miles").cast("decimal(6,2)"))
        .withColumn("days_on_market", F.col("days_on_market").cast("int"))
        # Cast dates
        .withColumn("sale_date", F.to_date(F.col("sale_date"), "yyyy-MM-dd"))
        # Cast booleans
        .withColumn("has_pool", F.col("has_pool").cast("boolean"))
        .withColumn("recently_renovated", F.col("recently_renovated").cast("boolean"))
        .withColumn("has_children", F.col("has_children").cast("boolean"))
        .withColumn("first_time_buyer", F.col("first_time_buyer").cast("boolean"))
    )
    return df


def transform(df: DataFrame) -> dict[str, DataFrame]:
    """Split the data by neighborhood and save each as a separate CSV file."""
    partitions = {}

    boolean_cols = [
        "has_pool",
        "recently_renovated",
        "has_children",
        "first_time_buyer",
    ]

    for hood in NEIGHBORHOODS:
        hood_df = df.filter(F.col("neighborhood") == hood)

        # Convert all boolean columns to "True"/"False" strings (ruff-safe)
        for col_name in boolean_cols:
            hood_df = hood_df.withColumn(
                col_name,
                F.when(F.col(col_name), "True").otherwise("False")
            )

        # Format distance_downtown_miles as integer for CSV consistency
        hood_df = hood_df.withColumn("distance_downtown_miles", F.col("distance_downtown_miles").cast("int"))

        # Save to CSV
        output_path = OUTPUT_FILES[hood]
        hood_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(output_path))

        partitions[hood] = hood_df

    return partitions


def load(partitions: dict[str, DataFrame], jdbc_url: str, pg_props: dict) -> None:
    """Insert each neighborhood dataset into its own PostgreSQL table."""
    for hood, hood_df in partitions.items():
        table_name = PG_TABLES[hood]

        hood_df.write \
            .jdbc(
                url=jdbc_url,
                table=table_name,
                mode="overwrite",
                properties=pg_props
            )

# ── Main (do not modify) ───────────────────────────────────────────────────────
def main() -> None:
    load_dotenv(ROOT / ".env")

    jdbc_url = (
        f"jdbc:postgresql://{os.getenv('PG_HOST', 'localhost')}:"
        f"{os.getenv('PG_PORT', '5432')}/{os.environ['PG_DATABASE']}"
    )
    pg_props = {
        "user":     os.environ["PG_USER"],
        "password": os.getenv("PG_PASSWORD", ""),
        "driver":   "org.postgresql.Driver",
    }
    csv_path = str(ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_FILE", "historical_purchases.csv"))

    spark = (
        SparkSession.builder.appName("HouseSaleETL")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df         = extract(spark, csv_path)
    partitions = transform(df)
    load(partitions, jdbc_url, pg_props)

    spark.stop()


if __name__ == "__main__":
    main()
