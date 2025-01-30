# Databricks notebook source
# MAGIC %md 
# MAGIC ## Correlations between data
# MAGIC
# MAGIC - Starting with the easiest correlation I can think of: average surface temperature and crop yield

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, MinMaxScaler, PCA, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation
from pyspark.mllib.linalg import Vectors
from pyspark.sql import functions as F

# COMMAND ----------

sql_query = """SELECT  cy.Year, cy.Country, cy.Item, cy.Element, cy.Unit, cy.Yield, avg_temp.Temperature
FROM agriculture_db.crop_yield cy
INNER JOIN agriculture_db.country_codes cc ON cc.ISONum = cy.AreaCodeM49
INNER JOIN 
(SELECT AVG(ast.Temperature) as Temperature, ast.Year, ast.ISO3
FROM agriculture_db.average_surface_temperature ast
GROUP BY ast.Country, ast.Year, ast.ISO3) avg_temp ON avg_temp.ISO3 = cc.ISO3 AND avg_temp.Year = cy.Year"""
features_df = spark.sql(sql_query)

# COMMAND ----------

features_df = features_df.withColumn("Yield",
                   F.when(F.isnull(F.col("Yield")), 0)
                   .otherwise(F.col("Yield").cast("int")))
features_df = features_df.withColumn("Temperature", F.round(F.col("Temperature"), 2))
features_df = features_df.where((features_df["Element"] == "Yield"))

# COMMAND ----------

features_df.show()

# COMMAND ----------

sns.set_theme(style="darkgrid")
df_plot = features_df.where((features_df["Country"] == "Germany") & (features_df["Item"] == "Wheat") & (features_df["Element"] == "Yield")).toPandas()
x_values = df_plot["Year"]
yields = df_plot["Yield"]
temperatures = df_plot["Temperature"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(x=x_values, y=yields, ax=axes[0])
axes[0].set_title("Yield Over Years")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Yield (kg/h)")
axes[0].set_yticks([yields.min(), yields.max()])

sns.lineplot(x=x_values, y=temperatures, ax=axes[1])
axes[1].set_title("Temperature Over Years")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_yticks([temperatures.min(), temperatures.max()])

plt.tight_layout()
plt.show()

# COMMAND ----------

assembler = VectorAssembler(inputCols=["Temperature", "Yield"], outputCol="features")
output = assembler.transform(features_df.filter((features_df["Country"] == "Germany") & (features_df["Item"] == "Wheat")))

# COMMAND ----------

correlation_matrix = Correlation.corr(output, "features", "pearson").head()[0]
r = correlation_matrix[0, 1]

print(r)

# COMMAND ----------

print(np.corrcoef(df_plot["Yield"].to_numpy(), df_plot["Temperature"].to_numpy()))

# COMMAND ----------

# MAGIC %md
# MAGIC - Having the Pearson correlation at 0.57 means there is a let's say medium correlation between the Yield and Average Surface Temperature
# MAGIC - This makes sense only by grouping together coutry, item and element, so maybe we can build a model that will have these as categorical data
# MAGIC - Firstly I am going to try a linear regression model

# COMMAND ----------

country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndexed")
item_indexer = StringIndexer(inputCol="Item", outputCol="ItemIndexed")

# COMMAND ----------

country_encoder = OneHotEncoder(inputCol="CountryIndexed", outputCol="CountryEncoded")
item_encoder = OneHotEncoder(inputCol="ItemIndexed", outputCol="ItemEncoded")

# COMMAND ----------

assembler = VectorAssembler(inputCols=["CountryEncoded", "ItemEncoded", "Temperature"], outputCol="features")

# COMMAND ----------

linear_regression_model = LinearRegression(featuresCol="features", labelCol="Yield")

# COMMAND ----------

pipeline = Pipeline(stages=[country_indexer, item_indexer, country_encoder, crop_encoder, assembler, linear_regression_model])
model = pipeline.fit(features_df)

# COMMAND ----------

predictions = model.transform(features_df)
predictions.select("Year", "Country", "Item", "Temperature", "Yield", "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Let's try to split the data and evaluate the model

# COMMAND ----------

train_data, test_data = features_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="Yield", predictionCol="prediction", metricName="rmse" )
rmse = evaluator.evaluate(predictions)
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² (Coefficient of Determination): {r2:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC - Having the country and item as encoded data along with the temperature, is not enough to predict the yield
# MAGIC - Let's try adding more features to our model

# COMMAND ----------

sql_query = """SELECT  cy.Year, cy.Country, cy.Item, cy.Element, cy.Unit, cy.Yield, avg_temp.Temperature, corr_index.CPI as CorruptionIndex, sog.GDP, up.UrbanPopulation, up.RuralPopulation
FROM agriculture_db.crop_yield cy
INNER JOIN agriculture_db.country_codes cc ON cc.ISONum = cy.AreaCodeM49
INNER JOIN 
(SELECT AVG(ast.Temperature) as Temperature, ast.Year, ast.ISO3
FROM agriculture_db.average_surface_temperature ast
GROUP BY ast.Country, ast.Year, ast.ISO3) avg_temp ON avg_temp.ISO3 = cc.ISO3 AND avg_temp.Year = cy.Year
INNER JOIN agriculture_db.corruption_index corr_index ON corr_index.ISO = cc.ISO3 AND corr_index.Year = cy.Year
INNER JOIN agriculture_db.share_of_gdp sog ON sog.Code = cc.ISO3 AND sog.Year = cy.Year
INNER JOIN agriculture_db.urban_population up ON up.Code = cc.ISO3 AND up.Year = cy.Year"""
features_df = spark.sql(sql_query)

# COMMAND ----------

features_df = features_df.withColumn("Yield",
                   F.when(F.isnull(F.col("Yield")), 0)
                   .otherwise(F.col("Yield").cast("int")))
features_df = features_df.withColumn("Temperature", F.round(F.col("Temperature"), 2)).withColumn("GDP", F.round(F.col("GDP"), 2))
features_df = features_df.where((features_df["Element"] == "Yield") & (features_df["CorruptionIndex"].isNotNull()))

# COMMAND ----------

features_df.show()

# COMMAND ----------

country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndexed")
item_indexer = StringIndexer(inputCol="Item", outputCol="ItemIndexed")
country_encoder = OneHotEncoder(inputCol="CountryIndexed", outputCol="CountryEncoded", dropLast=False)
item_encoder = OneHotEncoder(inputCol="ItemIndexed", outputCol="ItemEncoded", dropLast=False)

# COMMAND ----------

urban_population_assembler = VectorAssembler(inputCols=["UrbanPopulation"], outputCol="UrbanPopulationVectorized")
rural_population_assembler = VectorAssembler(inputCols=["RuralPopulation"], outputCol="RuralPopulationVectorized")
urban_population_scaler = MinMaxScaler(inputCol="UrbanPopulationVectorized", outputCol="UrbanPopulationScaled")
rural_population_scaler = MinMaxScaler(inputCol="RuralPopulationVectorized", outputCol="RuralPopulationScaled")

# COMMAND ----------

assembler = VectorAssembler(inputCols=["CountryEncoded", "ItemEncoded", "Temperature", "GDP", "CorruptionIndex", "UrbanPopulationScaled", "RuralPopulationScaled"], outputCol="features")

# COMMAND ----------

linear_regression_model = LinearRegression(featuresCol="features", labelCol="Yield")

# COMMAND ----------

pipeline = Pipeline(stages=[country_indexer, item_indexer, country_encoder, item_encoder, urban_population_assembler, rural_population_assembler, urban_population_scaler, rural_population_scaler, assembler, linear_regression_model])
model = pipeline.fit(features_df)

# COMMAND ----------

predictions = model.transform(features_df)
predictions.select("Year", "Country", "Item", "Temperature", "Yield", "prediction").show()

# COMMAND ----------

train_data, test_data = features_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="Yield", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² (Coefficient of Determination): {r2:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC - Our model got way better! This tells me maybe we need to consider more features in order to get better results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizing the data 
# MAGIC
# MAGIC - To gain insights from the different dimensions of this data, I would like to create some visualizations
# MAGIC - First I will only use the raw data
# MAGIC - Source article for different approaches: https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
# MAGIC
# MAGIC - The article suggests first using a **multivariate analysis**, which can be achieved by either a **pair-wise correlation matrix** or a **heatmap**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Heatmap plot
# MAGIC
# MAGIC Interpretation:
# MAGIC
# MAGIC Each square shows the correlation between the variables on each axis. Correlation ranges from -1 to +1
# MAGIC
# MAGIC Values closer to zero means there is no linear trend between the two variables
# MAGIC
# MAGIC The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other and the closer to 1 the stronger this relationship is
# MAGIC
# MAGIC A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases
# MAGIC
# MAGIC The diagonals are all 1/dark because those squares are correlating each variable to itself (so it's a perfect correlation)
# MAGIC
# MAGIC For the rest the larger the number and darker the color the higher the correlation between the two variables
# MAGIC
# MAGIC The plot is also symmetrical about the diagonal since the same two variables are being paired together in those squares

# COMMAND ----------

features_pd = features_df.select("Yield", "Temperature", "GDP", "CorruptionIndex", "UrbanPopulation", "RuralPopulation").toPandas()
fig, ax = plt.subplots(figsize=(10, 6))
corr = features_pd.corr()
heat_map = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt=".2f", linewidths=.05)
fig.subplots_adjust(top=0.93)
t= fig.suptitle("Attributes Correlation Heatmap", fontsize=14)

# COMMAND ----------

# MAGIC %md
# MAGIC - This is a heatmap for the data WITHOUT grouping together by year, country or item -> so we cannot really determine anything from this
# MAGIC - Let's try creating more heatmaps on grouped data

# COMMAND ----------

for feature in features_df.rdd.collect()[:5]:
    feature_pd = features_df.filter((features_df["Country"] == feature["Country"]) & (features_df["Item"] == feature["Item"])).select("Yield", "Temperature", "GDP", "CorruptionIndex", "UrbanPopulation", "RuralPopulation").toPandas()
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = feature_pd.corr()
    heat_map = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f', linewidths=.05)
    fig.subplots_adjust(top=0.93)
    t= fig.suptitle(f"Attributes Correlation Heatmap for {feature['Country']} {feature['Item']}", fontsize=14)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA visualization

# COMMAND ----------

pca_assembler = VectorAssembler(inputCols=["CountryEncoded", "ItemEncoded", "Temperature", "GDP", "CorruptionIndex", "UrbanPopulation", "RuralPopulation"], outputCol="features")
pca_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")

pca_pipeline = Pipeline(stages=[country_indexer, item_indexer, country_encoder, item_encoder, pca_assembler, pca_scaler, pca])
pca_model = pca_pipeline.fit(features_df)
pca_result = pca_model.transform(features_df)

# COMMAND ----------

pca_data = pca_result.select("pca_features", "Country").rdd.map(lambda row: (row.pca_features.toArray(), row.Country)).collect()
pca_pdf = pd.DataFrame([(pc[0], pc[1], country) for pc, country in pca_data], columns=["PC1", "PC2", "Country"])

# COMMAND ----------

country_pca = pca_pdf.groupby('Country')[['PC1', 'PC2']].mean().reset_index()

plt.figure(figsize=(12, 7))
scatter = plt.scatter(country_pca["PC1"], country_pca["PC2"], c=pd.factorize(country_pca["Country"])[0], cmap="tab20", alpha=0.6)

for country in country_pca["Country"]:
    country_data = country_pca[country_pca["Country"] == country]
    plt.scatter(country_data["PC1"], country_data["PC2"], label=country, alpha=0.7)
    plt.annotate(country, (country_data["PC1"].values[0], country_data["PC2"].values[0]), 
                fontsize=8, 
                alpha=0.7, 
                xytext=(5, 5),
                textcoords='offset points')

plt.title("PCA Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - We get this visualization with country and item categorical features as well
# MAGIC - I would like to see what happens if we remove them

# COMMAND ----------

pca_assembler = VectorAssembler(inputCols=["Temperature", "GDP", "CorruptionIndex", "UrbanPopulation", "RuralPopulation"], outputCol="features")
pca_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")

pca_pipeline = Pipeline(stages=[pca_assembler, pca_scaler, pca])
pca_model = pca_pipeline.fit(features_df)
pca_result = pca_model.transform(features_df)

# COMMAND ----------

pca_data = pca_result.select("pca_features", "Country").rdd.map(lambda row: (row.pca_features.toArray(), row.Country)).collect()
pca_pdf = pd.DataFrame([(pc[0], pc[1], country) for pc, country in pca_data], columns=["PC1", "PC2", "Country"])

# COMMAND ----------

country_pca = pca_pdf.groupby('Country')[['PC1', 'PC2']].mean().reset_index()

plt.figure(figsize=(12, 7))
scatter = plt.scatter(country_pca["PC1"], country_pca["PC2"], c=pd.factorize(country_pca["Country"])[0], cmap="tab20", alpha=0.6)

for country in country_pca["Country"]:
    country_data = country_pca[country_pca["Country"] == country]
    plt.scatter(country_data["PC1"], country_data["PC2"], label=country, alpha=0.7)
    plt.annotate(country, (country_data["PC1"].values[0], country_data["PC2"].values[0]), 
                fontsize=8, 
                alpha=0.7, 
                xytext=(5, 5),
                textcoords='offset points')

plt.title("PCA Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Even though the structure in the PCA is reversed, it does not have any significant changes in the structure: The outliers are still the same
# MAGIC - This means our categorical data is not that relevant, proving that the other features drive the most variance
# MAGIC - We could change the sign of the PCA so the images are the same if we want to

# COMMAND ----------

regression_assembler = VectorAssembler(inputCols=["pca_features"] + ["CountryEncoded", "ItemEncoded"], outputCol="final_features")
regression_model = LinearRegression(featuresCol="final_features", labelCol="Yield")
regression_pipeline = Pipeline(stages=[country_indexer, item_indexer, country_encoder, item_encoder, pca_assembler, pca_scaler, pca, regression_assembler, regression_model])
model = regression_pipeline.fit(features_df)

# COMMAND ----------

predictions = model.transform(features_df)
predictions.select("Year", "Country", "Item", "Temperature", "Yield", "prediction").show()

# COMMAND ----------

train_data, test_data = features_df.randomSplit([0.8, 0.2])
model = regression_pipeline.fit(train_data)
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="Yield", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² (Coefficient of Determination): {r2:.2f}")
