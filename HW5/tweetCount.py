import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import *
import json
import os

conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")


spark = SparkSession \
    .builder \
    .appName("Tweets and Country DF") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()



DF1 = spark.read.json("/home/rob/data/tweets.json")
DF1.show()

DF2 = spark.read.json("/home/rob/data/cityStateMap.json")
DF2.show()

DF3 = DF1.join(DF2, DF1.geo == DF2.city, 'inner').drop(DF2.city)
DF3.show()

DF4 = DF3.groupBy("state").count()
DF4.show()

DF4.write.json("TweetsOfEachState.jsonl")





# json_data = open("/home/rob/data/cityStateMap.json").read()
# cityMapList = json.loads(json_data)


# RDD1 = DF1.rdd
# RDD2 = RDD1.map(lambda attributes: Row(user=attributes[0], \
#                                        geo=attributes[1], \
#                                        tweet=attributes[2], \
#                                        state=pairCityandState(attributes[1])))
# DF2 = RDD2.toDF()
# DF2.show()

# DF2.write.json("/home/rob/data/joinedStateMap.jsonl")