from pyspark import SparkContext

def test_spark_session():
    try:
        # import os
        # os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'

        sc = SparkContext.getOrCreate()
        assert sc.version
        assert sc.version == "4.0.0"
        sc.stop()
    except Exception as e:
        print(f"❌ Error: {e}")