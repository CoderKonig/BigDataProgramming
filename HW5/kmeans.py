import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import numpy
import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession \
    .builder \
    .appName("K-means Algorithim") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Loads data
dataset = spark.read.format("libsvm").load("/home/rob/data/kmeans_input.txt")

# Trains a k-means model from the dataset
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


dataset.show()

label = dataset.select(dataset.columns[0])
label.show()

x = dataset.select(dataset.columns[0])
x.show()

pandasplot= dataset.toPandas()
from pandas.plotting import scatter_matrix

pandasplot(kind = 'scatter', x = 'x', y='label', colormap= 'winter_r')

# plt.scatter(x,y,c=labelcolumn)
# plt.show()