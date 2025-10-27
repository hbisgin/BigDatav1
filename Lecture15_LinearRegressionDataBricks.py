# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("lr_example").getOrCreate()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -P /tmp https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_linear_regression_data.txt
# MAGIC pwd

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/") #caution!

# COMMAND ----------

# Move file from local driver to DBFS
# Copy from local driver /tmp into DBFS
dbutils.fs.cp("file:/tmp/sample_linear_regression_data.txt", "dbfs:/BDA_UM_FLINT/datasets")

# COMMAND ----------

training = spark.read.format("libsvm").load("dbfs:/BDA_UM_FLINT/datasets/sample_linear_regression_data.txt")

# COMMAND ----------

training.count()

# COMMAND ----------

training.show()

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction")

# COMMAND ----------

lrModel = lr.fit(training)

# COMMAND ----------

lrModel.coefficients

# COMMAND ----------

lrModel.intercept

# COMMAND ----------

print("Coefficients: {}".format(str(lrModel.coefficients)))
print("\n")
print("Intercept: {}".format(str(lrModel.intercept)))

# COMMAND ----------

trainingSummary = lrModel.summary

# COMMAND ----------

trainingSummary.r2

# COMMAND ----------

trainingSummary.meanSquaredError

# COMMAND ----------

trainingSummary.rootMeanSquaredError

# COMMAND ----------

trainingSummary.residuals.show(10)
print("MAE: {}".format(trainingSummary.meanAbsoluteError))
print("MSE: {}".format(trainingSummary.meanSquaredError))
print("RMSE: {}".format(trainingSummary.rootMeanSquaredError))
print("r2: {}".format(trainingSummary.r2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and Test Split

# COMMAND ----------

all_data = spark.read.format("libsvm").load("dbfs:/BDA_UM_FLINT/datasets/sample_linear_regression_data.txt")

# COMMAND ----------

train_data, test_data = all_data.randomSplit([0.7,0.3])

# COMMAND ----------

train_data.show()

# COMMAND ----------

train_data.show(5)
train_data.describe().show()

# COMMAND ----------

test_data.show(5)
test_data.describe().show()

# COMMAND ----------

correct_model = lr.fit(train_data)

# COMMAND ----------

test_results = correct_model.evaluate(test_data)

# COMMAND ----------

test_results.residuals.show(5)
test_results.rootMeanSquaredError
print("RMSE on test data = %g" % test_results.rootMeanSquaredError)

# COMMAND ----------

unlabeled_data = test_data.select('features')
unlabeled_data.show()

# COMMAND ----------

predictions = correct_model.transform(unlabeled_data)

# COMMAND ----------

predictions.show()
