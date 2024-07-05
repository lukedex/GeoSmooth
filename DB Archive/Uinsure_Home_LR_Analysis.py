# Databricks notebook source
#Adds a column to the home table and designates it to df
from pyspark.sql.functions import *
import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,Factors to analyse - long list so collapsed for easy viewing.
columns_to_check=["Age_of_Property",
"Alarm_Ind",
"BDG_AD_Claims_last_5_Years",
"BDG_ADX_Claims_Last_5_Years",
"BDG_EOW_Claims_Last_5_Years",
"BDG_FI_Claims_Last_5_Years",
"BDG_FL_Claims_Last_5_Years",
"BDG_OT_Claims_Last_5_Years",
"BDG_ST_Claims_Last_5_Years",
"BDG_SU_Claims_Last_5_Years",
"BDG_TH_Claims_Last_5_Years",
"BDG_Years_Since_2nd_Claim",
"Bdgs_NCD_Years",
"Bike_SI_Band",
"Buildings_AD_Cover_Indicator",
"Buildings_Standard_Xs",
"Buildings_Sum_Insured",
"Buildings_Voluntary_Xs",
"Construction_Type",
"Contents_AD_Cover_Indicator",
"Contents_Standard_Xs",
"Contents_Sum_Insured",
"Contents_Voluntary_Xs",
"Cover_Type",
"Credit_Score",
"CT_AD_Claims_last_5_Years",
"CT_ADX_Claims_Last_5_Years",
"CT_EOW_Claims_Last_5_Years",
"CT_FI_Claims_Last_5_Years",
"CT_FL_Claims_Last_5_Years",
"CT_OT_Claims_Last_5_Years",
"CT_PP_Claims_Last_5_Years",
"CT_ST_Claims_Last_5_Years",
"CT_TH_Claims_Last_5_Years",
"CT_Years_Since_2nd_Claim",
"Cts_NCD_Years",
"Daily_Occupancy_Ind",
"Detailed_Occupation",
"Exposure_Year",
"HH_Software_House",
"Instalment_Ind",
"Legal_Ind",
"Marital_Sts",
"Nhood_Watch_Ind",
"No_Of_Occupants",
"Number_Of_Bedrooms",
"Occupation_Type",
"Oldest_Policyholder_Age",
"Ownership_Type",
"Policyholder_Sex",
"PP_Cover_Pedal_Cycle",
"PP_Cover_Specified",
"PP_Cover_Unspecified",
"Product_Group",
"Property_Type",
"Residency_Years",
"Smoker_Ind",
"Spec_SI_Band",
"Unspec_SI_Band"]

# COMMAND ----------

"""
Notes: Doesn't feel the most optimal way to loop through a SQL query but it does run fast. Only a 77k row table though.

If we could upload the final result into SQL I think we can move into SQL/Dashboard within DB for the rest of it.
"""

from pyspark.sql.types import StructType,StructField, StringType, FloatType


schema=StructType([
                    StructField('Exposure_year', FloatType(), True),
                    StructField('Rating_Factor', StringType(), True),
                    StructField('Factor_Level', StringType(), True),
                    StructField('NELR', FloatType(), True),
                    StructField('CAPPED_NELR', FloatType(), True),
                    StructField('AEP', FloatType(), True),
                    StructField('BurnCost', FloatType(), True),
                    StructField('CappedBurnCost', FloatType(), True),
                    StructField('BDG_Claim_Freq', FloatType(), True),
                    StructField('CT_Claim_Freq', FloatType(), True),
                    StructField('CT_Claim_Sev', FloatType(), True),
                    StructField('BDG_Claim_Sev', FloatType(), True),
                    StructField('EP_BDG', FloatType(), True),
                    StructField('EP_CT', FloatType(), True),
                    StructField('BDG_NELR', FloatType(), True),
                    StructField('CT_NELR', FloatType(), True),  
                    StructField('BDG_Capped_NELR', FloatType(), True),
                    StructField('CT_Capped_NELR', FloatType(), True),   
                    StructField('Commission', FloatType(), True),  
                    StructField('Exposure', FloatType(), True)
                    ])

spark_df1=spark.createDataFrame([],schema)

for factor in columns_to_check:
  qry = f"""
        select 
        Exposure_year,
        '{factor}' as Rating_Factor,
        {factor} as Factor_Level,
  
        sum(BDG_TOTAL_NIC+CT_TOTAL_NIC)/sum(NEP) as NELR,
        sum(BDG_TOTAL_CAPPED_NIC+CT_TOTAL_CAPPED_NIC)/sum(NEP) as CAPPED_NELR,
        
        sum(NEP)/sum(Exposure) as AEP,
        (sum(NEP)/sum(Exposure))*((sum(BDG_TOTAL_NIC)+sum(CT_TOTAL_NIC))/sum(NEP)) as BurnCost,
        (sum(NEP)/sum(Exposure))*((sum(BDG_TOTAL_CAPPED_NIC)+sum(CT_TOTAL_CAPPED_NIC))/sum(NEP)) as CappedBurnCost,
        
        sum(BDG_Claim_Count)/sum(Exposure) as BDG_Claim_Freq,
        sum(CT_Claim_Count)/sum(Exposure) as CT_Claim_Freq,
        sum(CT_Total_NIC)/sum(CT_Claim_Count) as CT_Claim_Sev,
        sum(BDG_Total_NIC)/sum(BDG_Claim_Count) as BDG_Claim_Sev,
        
        sum(EP_BDG) as EP_BDG,
        sum(EP_CT) as EP_CT,
        
        sum(BDG_Total_NIC)/sum(EP_BDG) as BDG_NELR,
        sum(CT_Total_NIC)/sum(EP_CT) as CT_NELR,
        sum(BDG_TOTAL_CAPPED_NIC)/sum(EP_BDG) as BDG_CAPPED_NELR,
        sum(CT_Total_CAPPED_NIC)/sum(EP_CT) as CT_CAPPED_NELR,        
        
        1-(sum(NEP)/sum(GEP)) as Commision,
        
        sum(exposure) as Exposure
        from pricing.uinsure_18_22_withcalcs
        group by Exposure_year, {factor}
        order by exposure_year, {factor}
  ;"""
  spark_df = spark.sql(qry)

  #Converts spark dataframe into pandas
  spark_df1=spark_df1.union(spark_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table pricing.uinsure_lr_summary;

# COMMAND ----------

spark_df1.write.mode("overwrite").saveAsTable("pricing.uinsure_lr_summary")
