# Databricks notebook source
# MAGIC %md
# MAGIC # Cleaning and preparing data 
# MAGIC
# MAGIC - This notebook processes the following datasets: **FAOSTAT data, Urbanization across the world, Share of GPD from agriculture, Religious composition by country, Global corruption index by country**
# MAGIC - Each dataset will have it's own Delta Table 

# COMMAND ----------

# MAGIC %md
# MAGIC ### FAOSTAT data
# MAGIC
# MAGIC - Source: **Food and Agriculture Organization of the United Nations** [FAOSTAT](https://www.fao.org/faostat/en/#data/QCL)
# MAGIC - The dataset contains data from **1960-2022**

# COMMAND ----------

import warnings

from pyspark.sql.functions import monotonically_increasing_id, regexp_replace, col, lit, format_number


warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# COMMAND ----------

storage_account_name = "cropyieldacc"
container_name = "agriculture-data"
sas_token = "sp=rl&st=2024-12-10T13:32:26Z&se=2025-03-31T20:32:26Z&spr=https&sv=2022-11-02&sr=c&sig=bBsDv78XRqyADtLW4Sx2RfEHpzgq9UKYrZTtlSKOCzw%3D"

blob_url = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/"
spark.conf.set(f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net", sas_token)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")

# COMMAND ----------

df = spark.read.csv(blob_url + "Production_Crops_Livestock_E_All_Data.csv", header=True, inferSchema=True)

# COMMAND ----------

df = df.drop("Item Code (CPC)")

# COMMAND ----------

transformed_columns = [
    regexp_replace(col(c), "[']", "").cast("int").alias(c) if c in ["Area Code (M49)"] else col(c)
    for c in df.columns
]
df = df.select(*transformed_columns)

# COMMAND ----------

df = df.withColumn("Id", monotonically_increasing_id())
columns = ["Id", "Area Code", "Area Code (M49)", "Area", "Item Code", "Item", "Element Code", "Element", "Unit"]

# COMMAND ----------

year_columns = [col for col in df.columns if col.startswith("Y") and not col.endswith(("F", "N"))]
flag_columns = [col for col in df.columns if col.endswith("F")]
note_columns = [col for col in df.columns if col.endswith("N")]

# COMMAND ----------

df_transform_year = df.melt(
    ids=[col for col in df.columns if col not in year_columns + list(flag_columns) + list(note_columns)], 
    values=year_columns,
    variableColumnName="Year", 
    valueColumnName="Yield"
)
df_transform_year = df_transform_year.withColumn("Final Year", regexp_replace(col("Year"), "Y", "").cast("int"))
df_transform_year = df_transform_year.drop("Year").withColumnRenamed("Final Year", "Year")

# COMMAND ----------

df_transform_flag = df.melt(
    ids=[col for col in df.columns if col not in year_columns + flag_columns + note_columns],
    values=flag_columns,
    variableColumnName="Year",
    valueColumnName="Year Flag"
)
df_transform_flag = df_transform_flag.withColumn("Final Year", regexp_replace(col("Year"), "Y|F", "").cast("int"))
df_transform_flag = df_transform_flag.drop("Year").withColumnRenamed("Final Year", "Year")

# COMMAND ----------

df_transform_note = df.melt(
    ids=[col for col in df.columns if col not in year_columns + flag_columns + note_columns],
    values=note_columns,
    variableColumnName="Year",
    valueColumnName="Year Note"
)
df_transform_note = df_transform_note.withColumn("Final Year", regexp_replace(col("Year"), "Y|N", "").cast("int"))
df_transform_note = df_transform_note.drop("Year").withColumnRenamed("Final Year", "Year")

# COMMAND ----------

df = df_transform_year \
    .join(df_transform_flag, on=[
        "Id", "Area Code", "Area Code (M49)", "Area", "Item Code", "Item", 
        "Element Code", "Element", "Unit", "Year"], how="left") \
    .join(df_transform_note, on=[
        "Id", "Area Code", "Area Code (M49)", "Area", "Item Code", "Item", 
        "Element Code", "Element", "Unit", "Year"], how="left")

# COMMAND ----------

cleaned_column_names = [col.replace(" ", "").replace("(", "").replace(")", "") for col in df.columns]
df = df.toDF(*cleaned_column_names)
df = df.withColumnRenamed("Area", "Country")

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.crop_yield")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Urbanization across the world
# MAGIC - Number of people living in rural areas by country 
# MAGIC - Source: https://ourworldindata.org/
# MAGIC - This dataset contains data from: **1960-2022**

# COMMAND ----------

df = spark.read.csv(f"{blob_url}Urban-And-Rural-Population-By-Country.csv", header=True, inferSchema=True)

# COMMAND ----------

df = df \
    .withColumnRenamed("Entity", "Country") \
    .withColumnRenamed("Urban population", "UrbanPopulation") \
    .withColumnRenamed("Rural population", "RuralPopulation")

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.urban_population")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Share of GPD from agriculture
# MAGIC
# MAGIC - Includes the percentage added to GPD from cultivation of crops and livestock production, as well as fishing, hunting and forestry
# MAGIC - Source: https://ourworldindata.org/grapher/agriculture-share-gdp?time=2022
# MAGIC - This dataset contains data from **1960-2022**

# COMMAND ----------

df = spark.read.csv(blob_url + "Share-Of-GDP-From-Agriculture.csv", header=True, inferSchema=True)
df.columns

# COMMAND ----------

df = df \
    .withColumnRenamed("Entity", "Country") \
    .withColumnRenamed("Agriculture, forestry, and fishing, value added (% of GDP)", "GDP")

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.share_of_gdp")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Religious composition by country
# MAGIC
# MAGIC - We know that Hinduism for example follows a vegetarian diet, based on this it's possible that livestock production for countries where Hinduism is popular might not be that high
# MAGIC - Source: https://www.pewresearch.org/religion/feature/religious-composition-by-country-2010-2050/
# MAGIC - This dataset contains data from **2010** and **2020**, additionally containing predictions for years 2030, 2040 and 2050
# MAGIC - Note: follow up on sources for dietary restrictions on different religions

# COMMAND ----------

import pandas as pd

from azure.storage.blob import BlobServiceClient
from io import BytesIO
from pyspark.sql import functions as F

# COMMAND ----------

blob_url_sc = f"https://{storage_account_name}.blob.core.windows.net/"
blob_service_client = BlobServiceClient(account_url=blob_url_sc, credential=sas_token)

# COMMAND ----------

blob_client = blob_service_client.get_container_client(container_name).get_blob_client("Religious_Composition_by_Country_2010-2050.xlsx")
blob = blob_client.download_blob().content_as_bytes()

# COMMAND ----------

df = pd.read_excel(BytesIO(blob), sheet_name="rounded_population")
df = df.drop(columns=["row_number", "level", "Nation_fk", "Region"])
df.head(2)

# COMMAND ----------

df = spark.createDataFrame(df)
df = df.filter(~col("Country").contains("All Countries"))

# COMMAND ----------

transformed_columns_population = [
    regexp_replace(col(c), "[<, ]", "").cast("bigint").alias(c) if c not in ["Year", "Country"] else col(c)
    for c in df.columns
]
df = df.select(*transformed_columns_population)

# COMMAND ----------

df = df.withColumnRenamed("Christians", "Christian") \
    .withColumnRenamed("Muslims", "Muslim") \
    .withColumnRenamed("Hindus", "Hindu") \
    .withColumnRenamed("Buddhists", "Buddhist") \
    .withColumnRenamed("Jews", "Jewish") \
    .withColumnRenamed("Folk Religions", "Folk") \
    .withColumnRenamed("Other Religions", "Other") \
    .withColumnRenamed("All Religions", "Population")

# COMMAND ----------

religions = ["Christian", "Muslim", "Hindu", "Buddhist", "Jewish", "Folk", "Other"]
df = df.select(
    "Year",
    "Country",
    "Population",
    F.explode(
        F.array(*[F.struct(F.lit(religion).alias("Religion"), F.col(religion).alias("Followers")) for religion in religions])
    ).alias("ReligionData")
).select("Year", "Country", "ReligionData.Religion", "ReligionData.Followers", "Population")

df.show(10)

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.religious_composition")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Global corruption index
# MAGIC
# MAGIC - The corruption index of each country
# MAGIC - Source: https://www.transparency.org/en/cpi/2022
# MAGIC - This dataset contains data from **1995-2022**
# MAGIC - For each year between 1995-2011 the data comes in separate .csv files, while data from 2012-2022 comes in one excel file

# COMMAND ----------

df = spark.read.csv(f"{blob_url}corruption-indexes/CPI-1995.csv", header=True, inferSchema=True)
df = df.withColumn("year", lit(1995))
df = df.select("country", "iso", "score", "year")

# COMMAND ----------

for year in range(1996, 2012):
    file_path = f"{blob_url}corruption-indexes/CPI-{year}.csv"
    current_df = spark.read.csv(file_path, header=True, inferSchema=True)
    current_df = current_df.withColumn("year", lit(year))
    current_df = current_df.select("country", "iso", "score", "year")
    df = df.union(current_df)

# COMMAND ----------

df = df.withColumn("score", (col("score") * 10).cast("int"))
df = df \
    .withColumnRenamed("country", "Country") \
    .withColumnRenamed("iso", "ISO") \
    .withColumnRenamed("score", "CPI") \
    .withColumnRenamed("year", "Year")

# COMMAND ----------

blob_client = blob_service_client.get_container_client(f"{container_name}/corruption-indexes").get_blob_client("CPI2012-2022.xlsx")
blob = blob_client.download_blob().content_as_bytes()

# COMMAND ----------

df_excel = pd.read_excel(BytesIO(blob), sheet_name="CPI Timeseries 2012 - 2022", header=2)
df_excel.head(5)

# COMMAND ----------

df_excel = spark.createDataFrame(df_excel)

# COMMAND ----------

df_excel = df_excel.select("Country / Territory", "ISO3", "CPI Score 2012", "CPI Score 2013", "CPI Score 2014", "CPI Score 2015", "CPI Score 2016", "CPI Score 2017", "CPI Score 2018", "CPI Score 2019", "CPI Score 2020", "CPI Score 2021", "CPI Score 2022")

# COMMAND ----------

cpi_columns = ["CPI Score 2012", "CPI Score 2013", "CPI Score 2014", "CPI Score 2015", "CPI Score 2016", "CPI Score 2017", "CPI Score 2018", "CPI Score 2019", "CPI Score 2020", "CPI Score 2021", "CPI Score 2022"]

# COMMAND ----------

df_excel = df_excel.select(
    *(c for c in df_excel.columns if c not in cpi_columns),
    *(col(c).cast("int").alias(c) for c in cpi_columns)
)

# COMMAND ----------

df_excel = df_excel.select(
    "Country / Territory", 
    "ISO3",
    F.explode(
        F.array(*[F.struct(F.lit(year).alias("Year"), F.col(col_name).alias("Score")) for year, col_name in zip(range(2012, 2023), cpi_columns)])
    ).alias("YearScore")
).select(
    F.col("Country / Territory").alias("Country"),
    F.col("ISO3").alias("ISO"),
    F.col("YearScore.Score").alias("CPI"),   
    F.col("YearScore.Year")
)

df_excel.show()

# COMMAND ----------

df_new = df.union(df_excel)
df_new.filter(df_new["Year"] == 1995).show()

# COMMAND ----------

df_new.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.corruption_index")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Change in global mean surface temperature caused by greenhouse gas emissions from agriculture and land use
# MAGIC - The global mean surface temperature change as a result of a country or region's cumulative emissions of carbon dioxide, methane, and nitrous oxide
# MAGIC - This is for land use and agriculture
# MAGIC only
# MAGIC - Source: [https://ourworldindata.org/grapher/global-warming-land?time=1895](https://ourworldindata.org/grapher/global-warming-land?time=1895)
# MAGIC - Contains data from years **1851** - **2023**

# COMMAND ----------

df = spark.read.csv(blob_url + "Contribution-from-greenhouse-gas-emissions.csv", header=True, inferSchema=True)

# COMMAND ----------

df = df.filter(df["Year"] > 1960) \
    .withColumnRenamed("Entity", "Country") \
    .withColumnRenamed("Code", "ISO3") \
    .withColumnRenamed("Change in global mean surface temperature caused by greenhouse gas emissions from agriculture and land use", "Contribution") #in Celcius
    
df = df.withColumn("Contribution", format_number("Contribution", 6))

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.temperature_change_by_greenhouse_emissions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average monthly surface temperature by country
# MAGIC
# MAGIC - The temperature of the air measured 2 meters above the ground, encompassing land, sea, and in-land water surfaces
# MAGIC - Source: [https://ourworldindata.org/grapher/average-monthly-surface-temperature?country=~CAN](url)
# MAGIC - Contains data from **1940** - **2024**

# COMMAND ----------

df = spark.read.csv(blob_url + "Average-Monthly-Surface-Temperature.csv", header=True, inferSchema=True)

# COMMAND ----------

df = df.drop("Average surface temperature5")
df = df.filter(df["Year"] > 1960) \
    .withColumnRenamed("Entity", "Country") \
    .withColumnRenamed("Code", "ISO3") \
    .withColumnRenamed("year", "Year") \
    .withColumnRenamed("Average surface temperature4", "Temperature")
df = df.withColumn("Temperature", format_number("Temperature", 2))

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.average_surface_temperature")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus
# MAGIC
# MAGIC - I think it will be useful along the way if we also had a table with the countries and their ISO codes, so I've created one
# MAGIC - Source of the data: https://www.kaggle.com/datasets/wbdill/country-codes-iso-3166?resource=download

# COMMAND ----------

df = spark.read.csv(f"{blob_url}Country-Codes.csv", header=True, inferSchema=True)

# COMMAND ----------

df = df.withColumnRenamed("iso2", "ISO2") \
    .withColumnRenamed("iso3", "ISO3") \
    .withColumnRenamed("iso_num", "ISONum") \
    .withColumnRenamed("country", "Country") \
    .withColumnRenamed("country_common", "CountryCommon")

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("agriculture_db.country_codes")
