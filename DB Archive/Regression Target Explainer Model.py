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
Dataset_SQL_Code = """select Member, age from
                        pricing.aa_silver_premiums_jc 
                        where Silver_InitialNetPremium > tiara_Covea_net"""

Target = "target_column"


# COMMAND ----------

# MAGIC %md
# MAGIC ## DO NOT EDIT FROM HERE ONWARDS ##
# MAGIC ______________________________________________________________________________________________________________
# MAGIC ###### Load the SQL query into a spark dataframe

# COMMAND ----------

spark_df = spark.sql(Dataset_SQL_Code)
full_df=spark_df.toPandas()

# COMMAND ----------

X_train=filtered_df[filtered_df.columns.drop(Target)]
y_train=filtered_df[Target]

# COMMAND ----------

# MAGIC %md
# MAGIC ######Drops all rows where the target is null

# COMMAND ----------

X_train_fix=X_train[y_train.isna()==False]
y_train_fix=y_train[y_train.isna()==False]
print(f'Train dataset shape (cols,rows): {X_train_fix.shape}')

# COMMAND ----------

!pip install interpret

# COMMAND ----------

from interpret.glassbox import ExplainableBoostingRegressor
ebm = ExplainableBoostingRegressor()
ebm.fit(X_train_fix, y_train_fix)

# COMMAND ----------

from interpret import show
ebm_global = ebm.explain_global()
show(ebm_global)
