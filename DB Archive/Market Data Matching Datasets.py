# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ###Simple code to compare 2 small datasets.
# MAGIC ##### I think if we use a different way of reading the tables this could run on more data. Using the current way is not efficient.

# COMMAND ----------

import pandas

# COMMAND ----------

spark_df = spark.sql("""select *, case when UUID is null then 1 else 0 end as match_ind from (
                            select *, row_number() over(partition by abicode, postcodekept, dobiso, avgprice1_5, paymenttype, volexcess, proposer_ncb, classofuse order by quotedatetimeiso desc ) as dedupe_num from pricing.dataset_to_check_temp_LD
                            --where uuid is not null
                            order by abicode, postcodekept, dobiso)
                        where dedupe_num=1""")
full_df=spark_df.toPandas()

# COMMAND ----------

remove_cols=[
'abicode'
,'createdat'
,'customerid'
,'dayofweekofbirth'
,'postcodekept'
,'quotedatetime'
,'quoteid'
,'randomid'
,'updatedat'
,'yearofquote'
,'day'
,'year'
,'dobiso'
,'quotedatetimeiso'
,'ukresidentsinceiso'
,'coverstartdateiso'
,'createdatiso'
,'updatedatiso'
,'ingestedatiso'
,'UUID'
,'covea_abicode'
,'fullPostcode'
,'NBQuoteCreatedDate'
,'policyInceptionDate'
,'transactionStartDate'
,'hub_timestamp'
,'dateOfBirth'
]
print(full_df.shape)
full_df=full_df[full_df.columns.drop(remove_cols)]
print(full_df.shape)
full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Gaming')))]
print(full_df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Defining the splits of data to compare. 
# MAGIC Calling them train and test, as this is layover from checking modelling datasets.

# COMMAND ----------

train_df = full_df[full_df['match_ind']==1]
test_df = full_df[full_df['match_ind']==0]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Removing any columns which would be constant for either test or train dataframe. Was running into mis-match formatting issues. Hoping this will resolve..

# COMMAND ----------

constasnt_cols_in_either=[]
for col in train_df.columns:
    if len(train_df[col].value_counts())<2 or len(test_df[col].value_counts())<2:
        constasnt_cols_in_either.append(col)

constasnt_cols_in_either.remove("match_ind")
non_constant_full_df=full_df[full_df.columns.drop(constasnt_cols_in_either)]
print(non_constant_full_df.shape)
print(f'List of columns removed are {constasnt_cols_in_either}. This could be worth checking!')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Assessing the dtypes are correct and correcting numerical ones.
# MAGIC This should throw an error if a column can't be converted

# COMMAND ----------

#non_constant_full_df.dtypes
import pandas as pd
cols_to_numeric=[
'carvalue'
,'dob'
#,'propftbusinesstype'
#,'proposer_fulltimeoccupation'
,'proposer_licencelengthmonths'
,'quotesinlast30days'
,'quotesinlastyear'
,'ukresidentsince'
,'yearofmanufacture'
#,'proposer_parttimeoccupation'
,'AvgPrice1_5'
,'avgexgroupprice1_5'
,'number_prices_returned'
,'avgnontelematics1_5'
,'AvgPrice6_10'
,'avgexgroupprice6_10']
non_constant_full_df[cols_to_numeric] = non_constant_full_df[cols_to_numeric].apply(pd.to_numeric)


# COMMAND ----------

non_constant_full_df.dtypes

# COMMAND ----------


train_df = non_constant_full_df[non_constant_full_df['match_ind']==1]
test_df = non_constant_full_df[non_constant_full_df['match_ind']==0]

print(f'Dataset 1 is shape {train_df.shape}, dataset 2 is shape {test_df.shape}')

# COMMAND ----------

#Optional - make both datasets the same size.

smallest_dataset_size = min(len(train_df),len(test_df))
trimmed_train_df = train_df.head(smallest_dataset_size)
trimmed_test_df = test_df.head(smallest_dataset_size)

print(f'Trimmed : \n Dataset 1 is now shape {trimmed_train_df.shape}, dataset 2 is now shape {trimmed_test_df.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### For Comparing 2 datasets
# MAGIC
# MAGIC If this is taking a while to run (over 10 minutes) it could be that your cluster is not appropriate. 

# COMMAND ----------

!pip install ydata-profiling==4.1.2
!pip install pydantic==1.10.2

# COMMAND ----------

from ydata_profiling import ProfileReport

train_report = ProfileReport(trimmed_train_df, title="No Matches", minimal=True, )
#train_report.to_file("train_report.html")

test_report = ProfileReport(trimmed_test_df, title="Joins Correctly", minimal=True)
#test_report.to_file("test_report.html")

comparison_report = train_report.compare(test_report)
comparison_report

# COMMAND ----------

# have missings go to numerical for numericals.. strings for strings?

# COMMAND ----------

from scipy.stats import kstest
#Duplicate dataset
filled_train=trimmed_train_df
filled_test=trimmed_test_df
#fill in missing numerical columns with -1
filled_train[cols_to_numeric] = trimmed_train_df[cols_to_numeric].fillna(-1)
filled_test[cols_to_numeric] = trimmed_test_df[cols_to_numeric].fillna(-1)
#fill rest in with strings
filled_train = trimmed_train_df.fillna('MISSING')
filled_test = trimmed_test_df.fillna('MISSING')


results_year=pd.DataFrame(columns=['Factor', 'Statistic', 'pvalue'])
for col in trimmed_train_df.columns.drop("match_ind"):
    _result=kstest(filled_train[col],filled_test[col])
    new_row = pd.DataFrame({'Factor':col, 'Statistic':_result.statistic, 'pvalue':_result.pvalue}, index=[0])
    results_year=pd.concat([new_row,results_year.loc[:]]).reset_index(drop=True)
results_year.sort_values('Statistic', ascending=False)

# COMMAND ----------


