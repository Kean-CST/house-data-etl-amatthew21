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
    return spark.read.csv(csv_path, header=True, inferSchema=False)


def transform(df: DataFrame) -> dict[str, DataFrame]:
    """Split the data by neighborhood and save each as a separate CSV file."""
    import shutil
    import glob

    # Boolean columns to normalise: raw CSV has TRUE/FALSE, expected output is True/False
    bool_cols = ["has_pool", "recently_renovated", "has_children", "first_time_buyer"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    partitions: dict[str, DataFrame] = {}
    for hood in NEIGHBORHOODS:
        hood_df = (
            df.filter(F.col("neighborhood") == hood)
            .orderBy("house_id")
        )
        partitions[hood] = hood_df 

        csv_df = hood_df
        for col in bool_cols:
            csv_df = csv_df.withColumn(
                col,
                F.when(F.upper(F.col(col)) == "TRUE", "True")
                 .when(F.upper(F.col(col)) == "FALSE", "False")
                 .otherwise(F.col(col))
            )
        csv_df = csv_df.withColumn(
            "sale_date",
            F.date_format(F.to_date(F.col("sale_date"), "M/d/yy"), "yyyy-MM-dd")
        )

        tmp_dir = str(OUTPUT_DIR / f"_tmp_{hood.replace(' ', '_').lower()}")
        csv_df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")
        part_files = glob.glob(f"{tmp_dir}/part-*.csv")
        import os as _os
        _os.replace(part_files[0], str(OUTPUT_FILES[hood]))
        shutil.rmtree(tmp_dir)

    return partitions


def load(partitions: dict[str, DataFrame], jdbc_url: str, pg_props: dict) -> None:
    """Insert each neighborhood dataset into its own PostgreSQL table."""
    cast_exprs = [
        F.col("house_id"),
        F.col("neighborhood"),
        F.col("price").cast("int"),
        F.col("square_feet").cast("int"),
        F.col("num_bedrooms").cast("int"),
        F.col("num_bathrooms").cast("int"),
        F.col("house_age").cast("int"),
        F.col("garage_spaces").cast("int"),
        F.col("lot_size_acres").cast("decimal(6,2)"),
        F.col("has_pool").cast("boolean"),
        F.col("recently_renovated").cast("boolean"),
        F.col("energy_rating"),
        F.col("location_score").cast("int"),
        F.col("school_rating").cast("int"),
        F.col("crime_rate").cast("int"),
        F.col("distance_downtown_miles").cast("decimal(6,2)"),
        F.to_date(F.col("sale_date"), "M/d/yy").alias("sale_date"),
        F.col("days_on_market").cast("int"),
    ]
    for hood, hood_df in partitions.items():
        table = PG_TABLES[hood]
        (
            hood_df.select(cast_exprs)
            .write.jdbc(url=jdbc_url, table=table, mode="overwrite", properties=pg_props)
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
