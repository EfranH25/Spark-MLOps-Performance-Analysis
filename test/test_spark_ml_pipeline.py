import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from services.spark_ml_pipeline import preprocess
import numpy as np


def _schema():
    # Minimal schema with all required feature columns + Amount + Class
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    fields = [
        StructField(c, DoubleType() if c != "Class" else IntegerType(), True)
        for c in feature_cols
    ]
    return StructType(fields)


def _tiny_rows():
    return [
        {
            **{f"V{i}": float(i) for i in range(1, 29)},
            "Amount": 10.0,
            "Class": 0,
            "Time": 1.0,
        },
        {
            **{f"V{i}": float(i * 2) for i in range(1, 29)},
            "Amount": 20.0,
            "Class": 0,
            "Time": 1.0,
        },
        {
            **{f"V{i}": float(i * 3) for i in range(1, 29)},
            "Amount": 30.0,
            "Class": 1,
            "Time": 1.0,
        },
    ]


def test_preprocess_adds_expected_columns(spark):
    df = spark.createDataFrame(_tiny_rows(), schema=_schema())
    scaled = preprocess(df)

    # Columns created by the pipeline
    assert "features" in scaled.columns
    assert "features_scaled" in scaled.columns

    # Sanity: features_scaled is a vector and has no nulls
    row = scaled.select("features_scaled").first()
    assert row[0] is not None
    assert (
        scaled.select(
            F.count(F.when(F.col("features_scaled").isNull(), 1)).alias("n")
        ).first()["n"]
        == 0
    )


def test_preprocess_check_features(spark):
    df = spark.read.csv(r"mock_data/test_creditcard.csv", header=True, inferSchema=True)
    scaled = preprocess(df)

    feature_cols = df.columns[:]

    # Columns created by the pipeline
    assert "features" in scaled.columns
    assert "features_scaled" in scaled.columns

    for orig, vec in zip(df.collect(), scaled.collect()):
        # Note Convert Row to list and compare to features vector
        orig_values = [orig[c] for c in feature_cols]
        features_vector = vec["features"].toArray().tolist()
        assert np.allclose(orig_values, features_vector)

        # ? NOTE: Can probably use numpy to validate the columns were processed right
