# Databricks notebook source
# MAGIC %md
# MAGIC #### This notebook will take 2 datasets (defined by a SQL query) and compare them to eachother with an interactive report.
# MAGIC ##### Please ensure the datasets brought in only have the columns you'd like to compare (i.e do not bring in something like UUID).
# MAGIC ##### All inputs to this script are in the cell below. Once you have edited this cell go to the top right of this screen and click 'Run All'. 

# COMMAND ----------

# Define your sql statements to run for each dataset. If you are not using 3 datasets then put 'Dataset_3_SQL_Code = None'.
Dataset_1_SQL_Code = """select Member, age from
pricing.aa_silver_premiums_jc where Silver_InitialNetPremium > tiara_Covea_net"""
Dataset_2_SQL_Code = """select Member, age from
pricing.aa_silver_premiums_jc where Silver_InitialNetPremium < tiara_Covea_net"""
Dataset_3_SQL_Code = """select Member, age from
pricing.aa_silver_premiums_jc where Silver_InitialNetPremium = tiara_Covea_net"""

#Define a title for each dataset. This is used in the report.
Dataset_1_Title = "Title 1"
Dataset_2_Title = "Title 2"
Dataset_3_Title = "Title 3"

# To enable better interpretation of graphs you can set both datasets to be equal population. 
# True or False is needed.

equalise_datasets = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## ----------DO NOT EDIT FROM HERE ONWARDS---------- ##
# MAGIC ###### Load the SQL query into a pandas dataframe

# COMMAND ----------

import pandas as pd

#Load dataframes as spark DF.
Dataset_1 = spark.sql(Dataset_1_SQL_Code)
Dataset_2 = spark.sql(Dataset_2_SQL_Code)
Dataset_3 = spark.sql(Dataset_2_SQL_Code)

# - In future the package may get an update to support spark for comparing. For now we must convert them to pandas dataframes.
df_1=Dataset_1.toPandas()
df_2=Dataset_2.toPandas()
df_3=Dataset_2.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Removing any columns which would be constant for either test or train dataframe. Was running into mis-match formatting issues. Hoping this will resolve..

# COMMAND ----------

constant_cols_in_either_dataset=[]
for col in df_1.columns:
    if len(df_1[col].value_counts())==1 or len(df_2[col].value_counts())==1 or len(df_3[col].value_counts())==1:
        constant_cols_in_either_dataset.append(col)

df_1=df_1[df_1.columns.drop(constant_cols_in_either_dataset)]
df_2=df_2[df_2.columns.drop(constant_cols_in_either_dataset)]
df_3=df_3[df_3.columns.drop(constant_cols_in_either_dataset)]

print('We have dropped the following columns due to them being constant in either the first or second dataset. \n', constant_cols_in_either_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Make both datasets the same size. This makes the graphs in the report match up so it's easier to spot differences.

# COMMAND ----------

if equalise_datasets == True:
    smallest_dataset_size = min(len(df_1),len(df_2),len(df_3))
    print(f'Before trimming: Dataset_1 had {len(df_1)} rows. Dataset_2 had {len(df_2)} rows. Dataset_3 had {len(df_3)} rows.')
    df_1 = df_1.head(smallest_dataset_size)
    df_2 = df_2.head(smallest_dataset_size)
    df_3 = df_3.head(smallest_dataset_size)
    print(f'After trimming: Dataset_1 has {len(df_1)} rows. Dataset_2 has {len(df_2)} rows. Dataset_3 has {len(df_3)} rows.')
else:
    print(f' Dataset_1 has {len(df_1)} rows. Dataset_2 has {len(df_2)} rows. Dataset_3 has {len(df_3)} rows.  No equalising has been done.')

# COMMAND ----------

# MAGIC %md
# MAGIC Distribution Analysis: This will conduct a goodness of fit test to show the statistical dispersion in distribution between the two datasets.

# COMMAND ----------

from scipy.stats import kstest
import numpy as np


#Duplicate datasets to ensure we don't mess with the graphical outputs.
kstest_df_1=df_1
kstest_df_2=df_2
kstest_df_3=df_3

#Finds a list of numerical columns in each dataset
numerical_columns_df1 = list(df_1.select_dtypes(include=[np.number]).columns.values)
numerical_columns_df2 = list(df_2.select_dtypes(include=[np.number]).columns.values)
numerical_columns_df3 = list(df_3.select_dtypes(include=[np.number]).columns.values)

#Checks if the datasets have different columns as numeric
if (len(numerical_columns_df1) - len(numerical_columns_df2)) + (len(numerical_columns_df1) - len(numerical_columns_df3)) != 0:
    print('Different number of numerical columns detected between the 3 datasets. Please check format of columns. Only columns found as numerical in all datasets shall be filled with -1 as missing.')

numerical_columns_in_all = list(set(numerical_columns_df1) & set(numerical_columns_df2) & set(numerical_columns_df3))

print('Numerical columns are:', numerical_columns_in_all)

#fill in missing numerical columns with -1
kstest_df_1[numerical_columns_in_all] = kstest_df_1[numerical_columns_in_all].fillna(-1)
kstest_df_2[numerical_columns_in_all] = kstest_df_2[numerical_columns_in_all].fillna(-1)
kstest_df_3[numerical_columns_in_all] = kstest_df_3[numerical_columns_in_all].fillna(-1)

#fill rest in with strings
kstest_df_1 = kstest_df_1.fillna('MISSING')
kstest_df_2 = kstest_df_2.fillna('MISSING')
kstest_df_3 = kstest_df_3.fillna('MISSING')

# COMMAND ----------

results_df=pd.DataFrame(columns=['Factor', 'Statistic', 'pvalue'])
for col in kstest_df_1.columns:
    _result=kstest(kstest_df_1[col],kstest_df_2[col])
    new_row = pd.DataFrame({'Factor':col, 'Statistic':_result.statistic, 'pvalue':_result.pvalue}, index=[0])
    results_df=pd.concat([new_row,results_df.loc[:]]).reset_index(drop=True)
print(f'Difference in distribution for {Dataset_1_Title} & {Dataset_2_Title}')
results_df.sort_values('Statistic', ascending=False)

# COMMAND ----------


results_df=pd.DataFrame(columns=['Factor', 'Statistic', 'pvalue'])
for col in kstest_df_2.columns:
    _result=kstest(kstest_df_2[col],kstest_df_3[col])
    new_row = pd.DataFrame({'Factor':col, 'Statistic':_result.statistic, 'pvalue':_result.pvalue}, index=[0])
    results_df=pd.concat([new_row,results_df.loc[:]]).reset_index(drop=True)
print(f'Difference in distribution for {Dataset_2_Title} & {Dataset_3_Title}')
results_df.sort_values('Statistic', ascending=False)

# COMMAND ----------


results_df=pd.DataFrame(columns=['Factor', 'Statistic', 'pvalue'])
for col in kstest_df_1.columns:
    _result=kstest(kstest_df_1[col],kstest_df_3[col])
    new_row = pd.DataFrame({'Factor':col, 'Statistic':_result.statistic, 'pvalue':_result.pvalue}, index=[0])
    results_df=pd.concat([new_row,results_df.loc[:]]).reset_index(drop=True)
print(f'Difference in distribution for {Dataset_1_Title} & {Dataset_3_Title}')
results_df.sort_values('Statistic', ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Depending on the cluster this cell may not be required. - Note: What happens if it already has package, does this error?

# COMMAND ----------

!pip install ydata-profiling==4.1.2
!pip install pydantic==1.10.2

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Main cell used to perform the data profiling and comparing the 2 data profiles.

# COMMAND ----------

from ydata_profiling import ProfileReport

dataset1_report = ProfileReport(df_1, title=Dataset_1_Title, minimal=True)
dataset2_report = ProfileReport(df_2, title=Dataset_2_Title, minimal=True)
comparison_report = dataset1_report.compare(dataset2_report)
comparison_report

# COMMAND ----------

###comparison_report.to_file("/dbfs/Users/Luke.Dexter@coveainsurance.co.uk/comparison.html")
