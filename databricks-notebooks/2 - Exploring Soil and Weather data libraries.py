# Databricks notebook source
# MAGIC %md
# MAGIC # Weather and Soil data
# MAGIC
# MAGIC - The two most important factors that can influence crop yield I assume are the weather and the type of soil that is present in a certain country
# MAGIC - Finding this sort of data proved to be quite a challenge, but luckily there are some Python libraries to work with

# COMMAND ----------

# MAGIC %md
# MAGIC ### Soil data 
# MAGIC
# MAGIC - The ISRIC World Soil Information website provides data about different soil profiles for each country
# MAGIC - Their recommendation of obtaining this data can be found here: https://www.isric.org/explore/soilgrids
# MAGIC - For Python they recommend using the [owslib](https://owslib.readthedocs.io/en/latest/) library, and we can find some [notebooks](https://git.wur.nl/isric/soilgrids/soilgrids.notebooks) with example usage

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather data
# MAGIC - There is a Python library that offers historical weather data that has among the data sources national weather services like the National Oceanic and Atmospheric Administration (NOAA) and Germany's national meteorological service (DWD)
# MAGIC - Documentation can be found here: https://dev.meteostat.net/python/#installation
# MAGIC - This contains data from **1973** - **2022**

# COMMAND ----------

import csv
import json
import gzip
import pandas as pd
import requests

from io import BytesIO, StringIO
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType

# COMMAND ----------

response = requests.get("https://bulk.meteostat.net/v2/stations/lite.json.gz")
with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
    stations_rdd = spark.sparkContext.parallelize(json.load(f)).map(lambda x: {"id": x["id"], "country": x["country"]})

# COMMAND ----------

df_stations = spark.createDataFrame(stations_rdd)

# COMMAND ----------

sql_query = """SELECT DISTINCT ISO2, ISONum, CountryCommon
FROM agriculture_db.country_codes cc
INNER JOIN agriculture_db.crop_yield on cc.ISONum = agriculture_db.crop_yield.AreaCodeM49"""

df_countries = spark.sql(sql_query)

# COMMAND ----------

df_country_station = df_stations.join(
    df_countries,
    df_stations["country"] == df_countries["ISO2"],
    "inner"
)

# COMMAND ----------

# df_country_weather = spark.createDataFrame([], schema=country_weather_schema)
# for df_country in df_countries.toLocalIterator():
#     station_ids = df_stations.filter(df_stations.country == df_country.ISO2).select("id").collect()
#     df_country_stations = spark.createDataFrame([], schema=schema)
#     for station_id in station_ids:
#         url = f"https://bulk.meteostat.net/v2/monthly/{station_id.id}.csv.gz"
#         try:
#             df_station = pd.read_csv(url, compression="gzip", names=columns, sep=",", header=None)
#             df_station = spark.createDataFrame(df_station)
#             df_country_stations = df_country_stations.union(df_station)
#         except Exception as e:
#             error_count += 1
#     df_country_stations = df_country_stations.groupBy("Year").agg(
#         F.round(F.avg("AverageTemperature"), 2).alias("AverageTemperature"),
#         F.min("MinimumTemperature").alias("MinimumTemperature"),
#         F.max("MaximumTemperature").alias("MaximumTemperature"),
#         F.round(F.sum("Precipitation"), 2).alias("Precipitation"),
#         F.round(F.sum("SnowDepth"), 2).alias("SnowDepth"),
#         F.round(F.avg("WindSpeed"), 2).alias("WindSpeed"),
#         F.max("PeakWindGust").alias("PeakWindGust"),
#         F.round(F.avg("SeaLevelPressure"), 2).alias("SeaLevelPressure"),
#         F.round(F.sum("SunshineDuration"), 2).alias("SunshineDuration")
#     ).orderBy("Year")
#     df_country_stations = df_country_stations.withColumn("Country", F.lit(df_country.CountryCommon)).withColumn("ISO2", F.lit(df_country.ISO2))
#     df_country_weather = df_country_weather.union(df_country_stations)

# COMMAND ----------

# def process_station(station_id: str, country_common: str, country_iso2: str, output_path: str):
#     url = f"https://bulk.meteostat.net/v2/daily/{station_id}.csv.gz"
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()

#         output = StringIO()
#         writer = csv.writer(output)
#         writer.writerow(columns)

#         with gzip.open(response.raw, mode="rt") as gz_file:
#             reader = csv.reader(gz_file)
#             for row in reader:
#                 writer.writerow(row)

#         csv_data = output.getvalue()
#         df_station = pd.read_csv(StringIO(csv_data), sep=",", header=0)
#         df_station = spark.createDataFrame(df_station) \
#             .withColumn("Year", F.year("Date")) \
#             .drop("Date") \
#             .filter((F.col("Year") > 1960) & (F.col("Year") < 2023)) \
#             .groupBy("Year") \
#             .agg(
#             F.round(F.avg("AverageTemperature"), 2).alias("AverageTemperature"),
#             F.min("MinimumTemperature").alias("MinimumTemperature"),
#             F.max("MaximumTemperature").alias("MaximumTemperature"),
#             F.round(F.sum("Precipitation"), 2).alias("Precipitation"),
#             F.round(F.sum("SnowDepth"), 2).alias("SnowDepth"),
#             F.round(F.avg("WindSpeed"), 2).alias("WindSpeed"),
#             F.max("PeakWindGust").alias("PeakWindGust"),
#             F.round(F.avg("SeaLevelPressure"), 2).alias("SeaLevelPressure"),
#             F.round(F.sum("SunshineDuration"), 2).alias("SunshineDuration"))  
#         df_station = df_station.withColumn("Country", F.lit(country_common)).withColumn("ISO2", F.lit(country_iso2))
#         df_station.write.mode("overwrite").parquet(output_path)
#     except Exception as e:
#         return None
# result = process_station("10637", "Germany", "DE", "/path")

# COMMAND ----------

# dbutils.fs.rm("/weather_results", recurse=True)
# output_dir = "/weather_results"
# dbutils.fs.mkdirs(output_dir)

# COMMAND ----------

columns = [
    "Date",                   # The date string (format: YYYY-MM-DD)
    "AverageTemperature",     # The average air temperature in °C, Float
    "MinimumTemperature",     # The average daily minimum air temperature in °C, Float
    "MaximumTemperature",     # The average daily maximum air temperature in °C, Float
    "Precipitation",          # The monthly precipitation total in mm, Float
    "SnowDepth",              # The maximum snow depth in mm, Integer
    "WindDirection",          # The average wind direction in degrees (°), Integer
    "WindSpeed",              # The average wind speed in km/h, Float
    "PeakWindGust",           # The peak wind gust in km/h, Float
    "SeaLevelPressure",       # The average sea-level air pressure in hPa, Float
    "SunshineDuration"        # The total monthly sunshine duration in hours, Float
]
schema = StructType([
    StructField("Year", IntegerType(), True),                  
    StructField("AverageTemperature", DoubleType(), True),     
    StructField("MinimumTemperature", DoubleType(), True),     
    StructField("MaximumTemperature", DoubleType(), True),    
    StructField("Precipitation", DoubleType(), True),
    StructField("SnowDepth", DoubleType(), True),
    StructField("WindSpeed", DoubleType(), True),             
    StructField("PeakWindGust", DoubleType(), True),           
    StructField("SeaLevelPressure", DoubleType(), True),        
    StructField("SunshineDuration", LongType(), True),
    StructField("Country", StringType(), True),
    StructField("ISO2", StringType(), True)      
])

# COMMAND ----------

def get_station_data(station_id):
    url = f"https://bulk.meteostat.net/v2/daily/{station_id}.csv.gz"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)

        with gzip.open(response.raw, mode="rt") as gz_file:
            reader = csv.reader(gz_file)
            for row in reader:
                writer.writerow(row)

        csv_data = output.getvalue()
        return csv_data
    except Exception as e:
        return None

# COMMAND ----------

csv_data = get_station_data("10637")
csv_df = pd.read_csv(StringIO(csv_data), sep=",", header=0)

results = []
        
csv_df["Year"] = pd.to_datetime(csv_df["Date"]).dt.year.astype(int)
csv_df = csv_df.drop(columns=["Date", "WindDirection"])
csv_df = csv_df[(csv_df["Year"] >= 1960) & (csv_df["Year"] <= 2022)]
csv_df = csv_df.groupby("Year").agg({
    "AverageTemperature": "mean",
    "MinimumTemperature": "min",
    "MaximumTemperature": "max",
    "Precipitation": "sum",
    "MaximumSnowDepth": "max",
    "WindSpeed": "mean",
    "PeakWindGust": "max",
    "SeaLevelPressure": "mean",
    "SunshineDuration": "sum"
}).reset_index()

for _, data in csv_df.iterrows():
    results.append(Row(**data.to_dict()))

# COMMAND ----------

def process_partition(partition):
    results = []
    
    for row in partition:
        csv_data = get_station_data(row["id"])
        if not csv_data:
            continue

        csv_df = pd.read_csv(StringIO(csv_data), sep=",", header=0)
        
        csv_df["Year"] = pd.to_datetime(csv_df["Date"]).dt.year
        csv_df = csv_df.drop(columns=["Date", "WindDirection"])
        csv_df = csv_df[(csv_df["Year"] >= 1960) & (csv_df["Year"] <= 2022)]
        csv_df = csv_df.groupby("Year").agg({
            "AverageTemperature": "mean",
            "MinimumTemperature": "min",
            "MaximumTemperature": "max",
            "Precipitation": "sum",
            "SnowDepth": "max",
            "WindSpeed": "mean",
            "PeakWindGust": "max",
            "SeaLevelPressure": "mean",
            "SunshineDuration": "sum"
        }).reset_index()

        csv_df["SunshineDuration"] = csv_df["SunshineDuration"].astype(int)
        csv_df["SnowDepth"] = csv_df["SnowDepth"].astype(float)
        csv_df["Country"] = row["CountryCommon"]
        csv_df["ISO2"] = row["ISO2"]

        for _, data in csv_df.iterrows():
            results.append(Row(**data.to_dict()))

    return results

# COMMAND ----------

df_country_station = df_country_station.repartition(100)
country_stations_rdd = df_country_station.rdd

country_stations_rdd = country_stations_rdd.mapPartitions(process_partition)

weather_df = spark.createDataFrame(country_stations_rdd, schema=schema)

# COMMAND ----------

weather_df.count()

# COMMAND ----------

weather_df = weather_df.groupBy("Year", "ISO2", "Country") \
            .agg(
            F.round(F.avg("AverageTemperature"), 2).alias("AverageTemperature"),
            F.min("MinimumTemperature").alias("MinimumTemperature"),
            F.max("MaximumTemperature").alias("MaximumTemperature"),
            F.round(F.sum("Precipitation"), 2).alias("Precipitation"),
            F.max("SnowDepth").alias("SnowDepth"),
            F.round(F.avg("WindSpeed"), 2).alias("WindSpeed"),
            F.max("PeakWindGust").alias("PeakWindGust"),
            F.round(F.avg("SeaLevelPressure"), 2).alias("SeaLevelPressure"),
            F.sum("SunshineDuration").alias("SunshineDuration")
            )

# COMMAND ----------

weather_df.count()

# COMMAND ----------

weather_df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.weather_data")
