# Databricks notebook source
# MAGIC %md
# MAGIC #### This notebook will take a dataset (defined by a SQL query) and create an interactive report profiling the data. Please ensure the dataset brought in only have the columns you'd like to profile (i.e do not bring in something like UUID).
# MAGIC ______________________________________________________________________________________________________________
# MAGIC ##### All inputs to this script are in the cell below. Once you have edited this cell go to the top right of this screen and click 'Run All'. 
# MAGIC
# MAGIC ______________________________________________________________________________________________________________
# MAGIC [Notes]: Requires a cluster with ML 13.1 on higher for the correct ydata_profiling to be installed. This function is also spark enabled.

# COMMAND ----------

# Define your sql statements to run. 
Dataset_SQL_Code = """select * from pricing.enrich_thatcham_abi_lookup"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## DO NOT EDIT FROM HERE ONWARDS ##
# MAGIC ______________________________________________________________________________________________________________
# MAGIC ###### Load the SQL query into a spark dataframe

# COMMAND ----------

import pandas as pd
#Load dataframes as spark DF.
Dataset_1 = spark.sql(Dataset_SQL_Code)

# COMMAND ----------

display(Dataset_1)

# COMMAND ----------

pandas_df=Dataset_1.toPandas()

# COMMAND ----------

!pip install ydata_profiling
!pip install typing_extensions
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Main cell used to perform the data profiling

# COMMAND ----------

from ydata_profiling import ProfileReport
df_profile = ProfileReport(pandas_df, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
df_profile
