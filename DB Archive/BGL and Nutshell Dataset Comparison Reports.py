# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ###Simple code to compare 2 small datasets.
# MAGIC ##### I think if we use a different way of reading the tables this could run on more data. Using the current way is not efficient.

# COMMAND ----------

import pandas

# COMMAND ----------

spark_df = spark.read.table("pricing.bgl_sold_quotes_2023_ld")
full_df=spark_df.toPandas()

# COMMAND ----------

remove_cols=['UUID',
'PolicyEffectiveDate',
'SchemeNumber',
'FullPostcode',
'RadarVersion',
'TransactionStartDate',
'NBQuoteCreatedDate',
'PolicyInceptionDate',
'PolarisVersion',
'BGLMOT',
'RACMileage',
'BGLAffinity'#,'BGLMandSCardHolder'
]
print(full_df.shape)
full_df=full_df[full_df.columns.drop(remove_cols)]
print(full_df.shape)
full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Gaming')))]
print(full_df.shape)

# COMMAND ----------

full_df=full_df.fillna(-1)

# COMMAND ----------

train_df = full_df[full_df['brand']=='Nutshell']
test_df = full_df[full_df['brand']=='BGL']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Removing any columns which would be constant for either test or train dataframe. Was running into mis-match formatting issues. Hoping this will resolve..

# COMMAND ----------

constasnt_cols_in_either=[]
for col in train_df.columns:
    if len(train_df[col].value_counts())==1 or len(test_df[col].value_counts())==1:
        constasnt_cols_in_either.append(col)

constasnt_cols_in_either.remove("brand")
full_df=full_df[full_df.columns.drop(constasnt_cols_in_either)]
print(full_df.shape)

# COMMAND ----------

Nb_only_df=full_df[full_df['TransactionType']=='NB']
train_df = Nb_only_df[Nb_only_df['brand']=='Nutshell']
test_df = Nb_only_df[Nb_only_df['brand']=='BGL']

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Comparing 2 datasets
# MAGIC
# MAGIC Takes a while to run.. even on less than 100k rows.

# COMMAND ----------

!pip install ydata-profiling==4.1.2
!pip install pydantic==1.10.2

# COMMAND ----------

from ydata_profiling import ProfileReport

comparison_report.config.html.style.theme = "flatly"

train_report = ProfileReport(train_df, title="BGL", minimal=True, )
#train_report.to_file("train_report.html")

test_report = ProfileReport(test_df, title="Nutshell", minimal=True)
#test_report.to_file("test_report.html")

comparison_report = train_report.compare(test_report)
comparison_report.config.html.style.theme = "Flatly"
comparison_report

# COMMAND ----------

dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)

# COMMAND ----------

comparison_report.to_file("/dbfs/Users/Luke.Dexter@coveainsurance.co.uk/comparison.html")

# COMMAND ----------

from scipy.stats import kstest

kstest(train_df['PDRCode'], test_df['PDRCode'])

# COMMAND ----------

train_df['PDRCode'].value_counts()

# COMMAND ----------

test_df['PDRCode'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ###For more than 2 datasets/periods to compare

# COMMAND ----------

"""
from ydata_profiling import ProfileReport, compare

comparison_report = compare([train_report, validation_report, test_report])

# Obtain merged statistics
statistics = comparison_report.get_description()

# Save report to file
comparison_report.to_file("comparison.html")
"""
