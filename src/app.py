from services import utils
from services import spark_ml_pipeline

import yaml

if __name__ == "__main__":
    print("starting")

    
    with open("../config/spark_ml_config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    if "credit_card_fraud" in config and config["credit_card_fraud"]["enable"]:
        data_output_folder = utils.get_data(
            data_handle=config["credit_card_fraud"]["data_handle"],
            data_folder=config["credit_card_fraud"]["data_folder"])

        if data_output_folder:
            print("START")
            spark_ml_pipeline.run(data_output_folder, **config["credit_card_fraud"]["job_config"])
            print("FINISHED")
    else:
        print("skipping credit card fraud detection.")
