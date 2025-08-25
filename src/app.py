import json
import logging
from os import path, listdir

from datetime import datetime

from services import utils
from services import spark_ml_pipeline

import yaml


def setup_logger(name: str, level: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    logger.info(f"Logging level: {logger.level}")
    return logger


def main():
    with open("../config/spark_ml_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    logger = setup_logger("app", config["log_level"])
    logger.info("Start")

    utils.setup_folders()
    logger.info("Setup folders")

    if "credit_card_fraud" in config and config["credit_card_fraud"]["enable"]:
        logger.info("Conducting Credit Card Fraud ML Pipeline")
        if config["credit_card_fraud"]["data_download"]:
            data_output_folder = utils.get_data(
                data_handle=config["credit_card_fraud"]["data_handle"],
                data_folder=config["credit_card_fraud"]["data_folder"],
            )
        else:
            if path.exists(
                path.join("data", config["credit_card_fraud"]["data_folder"])
            ) and listdir(
                path.join("data", config["credit_card_fraud"]["data_folder"])
            ):
                logger.info("Data folder with data already exists. Using data.")
                data_output_folder = path.join(
                    "data", config["credit_card_fraud"]["data_folder"]
                )
            else:
                raise f"'credit_card_fraud' called but no {config['credit_card_fraud']['data_folder']} folder or data found."
        if data_output_folder:
            logger.info("Starting credit card fraud clustering model")

            monitor = utils.ResourceMonitor(interval=1.0)
            monitor.start()
            metrics = {}
            try:
                metrics = spark_ml_pipeline.run(
                    data_output_folder, **config["credit_card_fraud"]["job_config"]
                )
            finally:
                monitor.stop()
                stats = monitor.get_stats()
                print(f"Peak CPU: {stats['peak_cpu_percent']}%")
                print(f"Peak Memory: {stats['peak_memory_mb']} MB")

                metrics["resource_stats"] = stats

            with open(
                path.join(
                    "metrics",
                    datetime.now().strftime("%H_%M_%S_")
                    + "credit_card_fraud_metrics.json",
                ),
                "w",
            ) as metrics_file:
                json.dump(metrics, metrics_file)

            logger.info("finished credit card fraud clustering model")
    else:
        logger.info("Skipping credit card fraud detection.")


# TODO: add hadoop
# TODO: try other spark ML models
# - Regression
# - Decision Tree
# TODO: Containerize training process to gather metrics
# Update README with report regarding how system effects metrics
# TODO: Deploy model and add FASTAPI to interact with models and pass in results. Maybe frontend UI
# TODO: [Optional] Fine Tune models further


if __name__ == "__main__":
    main()
