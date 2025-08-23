# Spark-MLOps-Performance-Analysis
This repo is to test and deploy the capabilities of SparkML

## 1. Data
1. download data from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. move `creditcard.csv` to `data/work`

## 2. Setup
1. run `pip install requirements.txt` to retrieve dependencies
2. in `src/spark_ml_pipeline.py` change `fraud_data` to the absolute file path of your `creditcard.csv` file 
   
    Note: Efran was having issues with relative file path with WSL so he used absolute file path instead
3. note your starting CPU & RAM utilization before running `spark_ml_pipeline.py`. for windows, `performance` in task manager should provide that info.
4. run `src/spark_ml_pipeline.py`. pay attention to your peak CPU and RAM usage.
5. fill out your table in https://docs.google.com/spreadsheets/d/1y7lUlPbpnDMeGuZjpiwNfq8jnuEklkyme1_E0cTZYTM/edit?usp=sharing

## 3. Running Hadoop
1. Followed these instructions: https://medium.com/@vikassharma555/hadoop-installation-on-windows-wsl-2-on-ubuntu-20-04-lts-single-node-d604729ea0ca
   - Note: This is for Ubuntu with WSL. All you need to do is install Hadoop and have it be running.
2. Run the following commands to make a directory for your data in Hadoop. 
   - `hdfs dfs -mkdir /cs532`
   - `hdfs dfs -mkdir /cs532/data`
3. Copy your `creditcard.csv` file (should be in `data/work/creditcard.csv`) to Hadoop
   - `hdfs dfs -copyFromLocal "/absolute/path/creditcard.csv" "/cs532/data"`
4. Run `src/spark_ml_pipeline_hadoop.py`
   - If you get `RPC response exceeds maximum data length` error make sure port in `hadoop_fraud_data` is correct. 
   - Based on the tutorial, it should be 9000. You can check `/usr/local/hadoop/etc/hadoop/core-site.xml` file where the correct url will be listed. For me its `hdfs://localhost:9000/`
5. Record data in table.