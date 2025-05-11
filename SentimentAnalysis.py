from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import NaiveBayes, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from utils import clean_text
import logging
from time import time
from datetime import datetime
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_file = os.path.join(log_dir, f"SentimentAnalysis_{timestamp}.log")
logging.basicConfig(filename=log_file,
                    format='%(asctime)s - %(message)s',
                    filemode="w",
                    level=logging.INFO)
logger = logging.getLogger()

spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "4G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

train_df = spark.read.option("header", "true").csv("./dataset/train.csv")
train_df = train_df.withColumn("label", (col("target") / 4).cast("int"))
clean_udf = udf(clean_text, StringType())
train_df = train_df.withColumn("clean_text", clean_udf(col("text")))

tokenizer = RegexTokenizer(pattern="\\W+", inputCol="clean_text", outputCol="tokens", gaps=True)
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_text")
hashingTF = HashingTF(numFeatures=5000, inputCol="filtered_text", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_text", outputCol="features")

preprocessing_start = time()

preprocessing_pipeline = Pipeline(stages=[tokenizer, remover, word2Vec])
model = preprocessing_pipeline.fit(train_df)
train_df = model.transform(train_df).repartition(128).cache()

preprocessing_time = time() - preprocessing_start
logger.info(f"Preprocessing Time: {preprocessing_time:.2f}s")

test_df = spark.read.option("header", "true").csv("./dataset/test.csv")
test_df = test_df.filter(test_df.target != "2")
test_df = test_df.withColumn("clean_text", clean_udf(col("text")))
test_df = test_df.withColumn("label", (col("target") / 4).cast("int"))
test_df = model.transform(test_df).repartition(2).cache()

# Naive Bayes #

# nb_start = time()

# modelType="multinomial"
# nb = NaiveBayes(featuresCol="features", labelCol="label", modelType=modelType)
# nb_model = nb.fit(train_df)
# predictions = nb_model.transform(test_df)
# accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# accuracy = accuracy_evaluator.evaluate(predictions)
# f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
# f1 = f1_evaluator.evaluate(predictions)

# nb_time = time() - nb_start
# logger.info(f"Naive Bayes (Type: {modelType}) Time: {nb_time:.2f}s - Accuracy: {accuracy:.2f} - F1 Score: {f1:.2f}")

# Logistic Regression #

lr_start = time()

lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)
predictions = lr_model.transform(test_df)
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = f1_evaluator.evaluate(predictions)

lr_time = time() - lr_start
logger.info(f"Logistic Regression Time: {lr_time:.2f}s - Accuracy: {accuracy:.2f} - F1 Score: {f1:.2f}")

# Random Forest #

rf_start = time()

numTress = 20
maxDepth = 5
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=numTress, maxDepth=maxDepth)
rfModel = rf.fit(train_df)
predictions = rfModel.transform(test_df)
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = f1_evaluator.evaluate(predictions)

rf_time = time() - rf_start
logger.info(f"Random Forest (Trees: {numTress} - Max Depth: {maxDepth}) Time: {rf_time:.2f}s - Accuracy: {accuracy:.2f} - F1 Score: {f1:.2f}")

spark.stop()