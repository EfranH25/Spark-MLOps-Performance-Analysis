import time
import statistics
from typing import List, Dict

# import math


import numpy as np
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession, functions
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

import logging

logger = logging.getLogger("app")

# setting random seed for notebook reproducibility
RND_SEED = 23
np.random.seed = RND_SEED
np.random.set_state = RND_SEED


def setup(path: str) -> (SparkSession, DataFrame):
    """
    Starts up spark session and loads fraud data into dataframe
    :param path: absolute path to fraud data
    :return: spark session and fraud data loaded in spark
    """
    spark: SparkSession = SparkSession.builder.appName(
        "CreditFraudDetector"
    ).getOrCreate()
    df: DataFrame = spark.read.csv(path, header=True, inferSchema=True)
    return spark, df


def preprocess(df: DataFrame) -> DataFrame:
    """
    Applies preprocessing step for ML pipeline
    :param df: spark df with credit card data loaded
    :return: dense vector with scaled data in "features_scaled" column
    """

    feature_cols = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
        "Class",
    ]

    # Move data to dense vector. This vectorization process transforms data into format that ML model is trained on.
    # We select the relevant features we want to train our model on. In this case, thats all the columns minus time.
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_df = assembler.transform(df)
    # assembled_df.show(10, truncate=False)

    # TODO: add additional preprocessing steps to improve performance
    # Standard scaler normalizes features to a common scale. Normalization puts values into similar scale which can
    # help ML training because data points are more like for like.
    standard_scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    scaled_df = standard_scaler.fit(assembled_df).transform(assembled_df)
    # scaled_df.select("features", "features_scaled").show(10, truncate=False)

    return scaled_df


def get_f1_score(predictions, model):
    """
    Gets f1 score of kmean ml model
    :param predictions: predictions from kmean model
    :return: f1 score
    """

    # Get centroids. This is the center of each cluster in the model. For each prediction, we will
    # calculate how far its distance is from the centroid point for the cluster they were assigned.
    centroids = model.clusterCenters()

    # Define a UDF to calculate Euclidean distance to the assigned cluster centroid
    def compute_distance(features, cluster):
        """
        Helper function for calculating distance from centroids in anomaly detection
        :param features: column name with scaled features
        :param cluster: column with nearest cluster
        :return: distance from cluster
        """
        centroid = centroids[cluster]
        return float(np.sqrt(np.sum((np.array(features) - np.array(centroid)) ** 2)))

    distance_udf = udf(compute_distance, returnType=FloatType())
    predictions = predictions.withColumn(
        "distance", distance_udf("features_scaled", "predictedClass")
    )

    # Calculate the threshold based on the distribution of distances. About 0.2% of our data is Fraud.
    # If a points distance falls in the 99.8% quantile, then it is an anomaly and flagged as Fraud.
    quantiles = predictions.approxQuantile("distance", [0.998], 0.001)
    threshold = quantiles[0]
    # Flag anomalies
    predictions = predictions.withColumn(
        "isAnomaly", functions.when(predictions["distance"] > threshold, 1).otherwise(0)
    )

    # Calculates teh F1 Score of predictions based on what was flagged as "isAnomaly" with its true class value.
    # Calculate TP, FP, FN for Precision, Recall, and F1 Score
    predictions = (
        predictions.withColumn(
            "TP",
            functions.when(
                (predictions["Class"] == 1) & (predictions["isAnomaly"] == 1), 1
            ).otherwise(0),
        )
        .withColumn(
            "FP",
            functions.when(
                (predictions["Class"] == 1) & (predictions["isAnomaly"] == 0), 1
            ).otherwise(0),
        )
        .withColumn(
            "FN",
            functions.when(
                (predictions["Class"] == 0) & (predictions["isAnomaly"] == 1), 1
            ).otherwise(0),
        )
    )

    # Aggregate TP, FP, and FN
    agg_df = predictions.agg(
        functions.sum("TP").alias("TP"),
        functions.sum("FP").alias("FP"),
        functions.sum("FN").alias("FN"),
    )

    # Calculate precision and recall
    result = agg_df.withColumn(
        "precision", agg_df["TP"] / (agg_df["TP"] + agg_df["FP"])
    ).withColumn("recall", agg_df["TP"] / (agg_df["TP"] + agg_df["FN"]))

    result = result.withColumn(
        "F1",
        (2 * result["precision"] * result["recall"])
        / (result["precision"] + result["recall"]),
    )

    return result.first()["F1"]


def ml_model(metrics: Dict, df: DataFrame, cluster_list: List[int], folds: int = 5):
    """
    Trains K-Means ml model for anomaly detection via pyspark
    :param metrics: Dictionary that contains metrics
    :param df: preprocessed spark df with credit card data
    :param cluster_list: list of cluster numbers
    :param folds: number of folds for cross validation
    :return: results of ml training
    """

    logger.info("Starting ml process")
    # We split our data into training and test set. We will train our model on the train set to pick our best model.
    # Then we see how our best model performs by evaluating it on our test set.
    # TODO: add functionality for stratified splits
    # https://stackoverflow.com/questions/47637760/stratified-sampling-with-pyspark
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=RND_SEED)

    # 5-Fold Cross Validation
    # We use the K-Means ML model for our anomaly detection. The K-Means model takes in the variable K to
    # determine the number of clusters to group our data. We do not know which K value is best. So we will
    # create a list of potential K values and use the average results from 5-Fold Cross Validation to determine the
    # best K value for our model
    results = {}
    results_metrics = {}

    for cluster in cluster_list:
        entry = {}
        cluster_time = time.time()
        time_list = []
        f1_score_list = []

        # Initialize and train a K-means model. "features_scaled" is our dense vector and "predictedClass" is the
        # cluster each credit card transaction belongs too.
        kmeans = KMeans(
            k=cluster, featuresCol="features_scaled", predictionCol="predictedClass"
        ).setSeed(1)

        # Perform 5-Fold Cross validation with K-Means model from cluster in cluster_list.
        # For each fold, a random 80/20 split of data is taken from the training set. The model clusters on
        # the 80 set and is scored on the 20 set. F1 score is used to score model accuracy. The F1 score for each
        # fold is stored for each K. The highest average F1 score determines the best K.
        for _ in range(folds):
            # TODO: add functionality for stratified splits
            fold_train_data, fold_test_data = train_data.randomSplit([0.8, 0.2])
            model = kmeans.fit(fold_train_data)
            predictions = model.transform(fold_test_data)

            # Calculate F1 Score
            f1_score = get_f1_score(predictions, model)
            f1_score_list.append(f1_score)
            time_list.append(time.time() - cluster_time)

        # Track results per cluster.
        entry["f1_scores"] = f1_score_list
        entry["mean_f1_score"] = statistics.mean(f1_score_list)
        entry["times"] = time_list
        entry["mean_time"] = statistics.mean(time_list)
        logger.info(f"Cluster {cluster} \n Entry: {entry}")
        results_metrics[cluster] = entry.copy()
        entry["model"] = kmeans
        results[cluster] = entry

    # logger.info(results)

    metrics["training_results"] = results_metrics

    best_f1 = -1
    best_model = None

    # Selects best model
    for cluster in results:
        if results[cluster]["mean_f1_score"] > best_f1:
            best_f1 = results[cluster]["mean_f1_score"]
            best_model = results[cluster]["model"]

    model = best_model.fit(train_data)
    model.save("spark_ml_pipeline.model")

    # Performs prediction with the best model on the test data. Returns F1 score
    start = time.time()
    predictions = model.transform(test_data)
    end = time.time()

    metrics["best_model_pred_time"] = (end - start) * 10**3
    logger.info(f"time for predicting test data: {metrics['best_model_pred_time']} ms")

    best_f1 = get_f1_score(predictions, model)
    metrics["best_f1_score"] = best_f1
    return best_f1, metrics


def run(data, cluster_list):
    """
    Runs application. Also times every major step of application
    """

    metrics = {}
    pipeline_start = time.time()

    # Step 1: Loading data
    # Initialize spark session and load credit card data from LFS or HDFS into Spark DataFrame
    # NOTE: replace with absolute file path for creditcard.csv or hadoop instance
    # fraud_data = "hdfs://localhost:9000/cs532/data/creditcard.csv"
    fraud_data = data

    start = time.time()
    spark, credit_df = setup(fraud_data)
    end = time.time()
    metrics["spark_setup_data_lost"] = (end - start) * 10**3

    logger.info(
        f"Execution time for setup of spark and loading data: {metrics['spark_setup_data_lost']} ms"
    )

    # Step 2. Preprocessing data
    # The preprocessing step vectorizes credit card data so that it can be ingested in our ML model for training
    # and prediction. For our preprocessing step, we put the training and prediction column into a dense vector and
    # then normalize our data.
    start = time.time()
    scaled_df = preprocess(credit_df)
    end = time.time()
    metrics["preprocessing"] = (end - start) * 10**3
    logger.info(f"Execution time for preprocessing data: {metrics['preprocessing']}ms")

    # Step 3: Building ML Model
    # Trains K-Means model with credit card transaction data. Uses 5-Folds Cross Validation to determine
    # best K for fraud detection. Selects best model based on F1 score. Runs best model on test set and returns
    # F1 score from best model to show how it performed.
    start = time.time()
    results, metrics = ml_model(metrics, scaled_df, cluster_list)
    logger.info(f"Best F1 Score: {results}")
    end = time.time()
    metrics["ml_training"] = (end - start) * 10**3

    logger.info(f"Execution time for training ML model: {metrics['ml_training']}ms")

    spark.stop()
    end = time.time()

    metrics["pipeline_runtime"] = (end - pipeline_start) * 10**3

    logger.info(f"Final execution time for ML process: {metrics["pipeline_runtime"]}ms")

    return metrics
