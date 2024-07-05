# Databricks notebook source
# MAGIC %md
# MAGIC This script will look at the combined manufacturer + model of cars which are showing trends in the amount of brokers returning a quote for them.

# COMMAND ----------

# MAGIC %md
# MAGIC Notes: Have this loop through a list of factors to check

# COMMAND ----------


def trend_checker(factor_name, minimum_quotes=50):

    temp_df=spark.sql("""with period_1 as  (
    select
    {x} as factor,
    avg(avgPrice15) as p1_price, 
    avg(numberpricesreturned) as p1_returned,
    count(*) as p1_count
    FROM gocompare.gold_jl_disc
    inner join gocompare.gold_shrd_risks using (quoteid)
    left outer join pricing.enrich_thatcham_abi_lookup enrich on abicode=abi_code
    where gocompare.gold_jl_disc.quotedatetime between current_date() - 17 and current_date() - 3
    and daysToInception < 3 
    group by all
    ),

        period_2 as  (
    select
    {x} as factor,
    avg(avgPrice15) as p2_price, 
    avg(numberpricesreturned) as p2_returned,
    count(*) as p2_count
    FROM gocompare.gold_jl_disc
    inner join gocompare.gold_shrd_risks using (quoteid)
    left outer join pricing.enrich_thatcham_abi_lookup enrich on abicode=abi_code
    where gocompare.gold_jl_disc.quotedatetime between current_date() - 24 and current_date() - 10
    and daysToInception < 3 
    group by all
    ),

        period_3 as  (
    select
    {x} as factor,
    avg(avgPrice15) as p3_price, 
    avg(numberpricesreturned) as p3_returned,
    count(*) as p3_count
    FROM gocompare.gold_jl_disc
    inner join gocompare.gold_shrd_risks using (quoteid)
    left outer join pricing.enrich_thatcham_abi_lookup enrich on abicode=abi_code
    where gocompare.gold_jl_disc.quotedatetime between current_date() - 31 and current_date() - 17
    and daysToInception < 3 
    group by all
    ),

        period_4 as  (
    select
    {x} as factor,
    avg(avgPrice15) as p4_price, 
    avg(numberpricesreturned) as p4_returned,
    count(*) as p4_count
    FROM gocompare.gold_jl_disc
    inner join gocompare.gold_shrd_risks using (quoteid)
    left outer join pricing.enrich_thatcham_abi_lookup enrich on abicode=abi_code
    where gocompare.gold_jl_disc.quotedatetime between current_date() - 38 and current_date() - 24
    and daysToInception < 3 
    group by all
    ),

        period_5 as  (
    select
    {x} as factor,
    avg(avgPrice15) as p5_price, 
    avg(numberpricesreturned) as p5_returned,
    count(*) as p5_count
    FROM gocompare.gold_jl_disc
    inner join gocompare.gold_shrd_risks using (quoteid)
    left outer join pricing.enrich_thatcham_abi_lookup enrich on abicode=abi_code
    where gocompare.gold_jl_disc.quotedatetime between current_date() - 45 and current_date() - 31
    and daysToInception < 3 
    group by all
    ),

        period_6 as  (
    select
    {x} as factor,
    avg(avgPrice15) as p6_price, 
    avg(numberpricesreturned) as p6_returned,
    count(*) as p6_count
    FROM gocompare.gold_jl_disc
    inner join gocompare.gold_shrd_risks using (quoteid)
    left outer join pricing.enrich_thatcham_abi_lookup enrich on abicode=abi_code
    where gocompare.gold_jl_disc.quotedatetime between current_date() - 52 and current_date() - 38
    and daysToInception < 3 
    group by all
    )

        select 
        factor as Factor_Level,
        p1_returned/((p2_returned+p3_returned+p4_returned+p5_returned)/4) as Last2Week_v_AvgMonth,
        p1_returned as P1_QteReturn,
        p2_returned as P2_QteReturn,
        p3_returned as P3_QteReturn,
        p4_returned as P4_QteReturn,
        p5_returned as P5_QteReturn,
        p6_returned as P6_QteReturn,
        p1_count as P1_QteVolume,
        p2_count as P2_QteVolume,
        p3_count as P3_QteVolume,
        p4_count as P4_QteVolume,
        p5_count as P5_QteVolume,
        p6_count as P6_QteVolume,
        p1_price as P1_AvgPrice,
        p2_price as P2_AvgPrice,
        p3_price as P3_AvgPrice,
        p4_price as P4_AvgPrice,
        p5_price as P5_AvgPrice,
        p6_price as P6_AvgPrice,
        p1_price/p2_price as rel_price_diff
        from period_1 
        inner join period_2 using (factor)
        inner join period_3 using (factor)
        inner join period_4 using (factor)
        inner join period_5 using (factor)
        inner join period_6 using (factor)
        where p1_count > {volume}
        and p2_count > {volume}
        order by last2week_v_avgmonth asc""".format(x=factor_name,volume=minimum_quotes)).toPandas()
    return temp_df

# COMMAND ----------

manumodel_df = trend_checker(factor_name='concat(manufacturer, \' \', model)',
                             minimum_quotes=200)
occupation_df = trend_checker(factor_name='proposerFullTimeOccupation',
                              minimum_quotes=200)
postcode_df = trend_checker(factor_name='substr(postcodekept,0,3)', 
                            minimum_quotes=200)
overall_df = trend_checker(factor_name='1', #Set to a constant to get view of all quotes
                            minimum_quotes=200)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graphs and outputs for dashboard

# COMMAND ----------

display(postcode_df)

# COMMAND ----------

display(occupation_df)

# COMMAND ----------

display(manumodel_df)

# COMMAND ----------

#Summary/Alerts Table

import pandas as pd

alerts_df = pd.concat([manumodel_df,occupation_df])
display(alerts_df[(alerts_df['Last2Week_v_AvgMonth'] < 0.8) | (alerts_df['Last2Week_v_AvgMonth'] > 1.2)])

# COMMAND ----------

overall_df

# COMMAND ----------

import datetime
import pandas as pd
# Create a overall DF which updates on every run to show new dates chosen.

time_periods_end_used_list=[
      f'{(datetime.datetime.now() - datetime.timedelta(days=3)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=10)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=17)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=24)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=31)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=38)).date()}']

time_periods_start_used_list=[
      f'{(datetime.datetime.now() - datetime.timedelta(days=17)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=24)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=31)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=38)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=45)).date()}',
      f'{(datetime.datetime.now() - datetime.timedelta(days=52)).date()}']

overall_dataset=pd.DataFrame(data=time_periods_end_used_list,columns=['Time_Period_End'])
overall_dataset['Time_Period_Start']=time_periods_start_used_list
overall_dataset['Avg_Quotes_Returned']=overall_df[['P1_QteReturn','P2_QteReturn','P3_QteReturn','P4_QteReturn','P5_QteReturn','P6_QteReturn']].transpose()[0].values
overall_dataset['Quote_Volumes']=overall_df[['P1_QteVolume','P2_QteVolume','P3_QteVolume','P4_QteVolume','P5_QteVolume','P6_QteVolume']].transpose()[0].values
overall_dataset['Avg_Price']=overall_df[['P1_AvgPrice','P2_AvgPrice','P3_AvgPrice','P4_AvgPrice','P5_AvgPrice','P6_AvgPrice']].transpose()[0].values
overall_dataset['Variable_Prefix']=['P1','P2','P3','P4','P5','P6']
display(overall_dataset)

# COMMAND ----------

#information_df = pd.DataFrame(data=time_periods_used_list,columns=['Time_Periods'])
#information_df['Variable_Prefix'] = ['p1','p2','p3','p4','p5','p6']

#display(information_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Information:
# MAGIC
# MAGIC 'Movement' is defined as the difference in the number of brokers returning a price for a quote in the last 2 weeks compared to the average of the 5 weeks preceeding it. 
# MAGIC Postcodes have been excluded from the summary table due to the amount of responses that fall within this criteria. 
# MAGIC
# MAGIC Filters applied to analysis: 
# MAGIC 1. Days to inception < 3 
# MAGIC 1. Quote Volume for segment < 200

# COMMAND ----------

# MAGIC %md
# MAGIC | Syntax      | Description |
# MAGIC | ----------- | ----------- |
# MAGIC | P1/P2/../P6      | Prefix indicator related to the time period used.   |
# MAGIC | QteReturn      | How many brokers returned a price for a single quote. |
# MAGIC | QteVolume   | The volume of quotes.        |
# MAGIC | AvgPrice   | The average price returned per quote.       |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Overall Market Analysis 
# MAGIC > Use these graphs to ensure the market has not moved too far from 1 time period to another. 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Alert Table
# MAGIC > This table highlights the segments of tracked factors which have shown movement beyond 20%.\
# MAGIC > Postcode District has been excluded from this graph due to the amount of entries it produces.
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Detailed insights per factor
# MAGIC > Tables showing all segments of a given factor and their performance by the defined metrics over time.
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC #### Additional Information:
# MAGIC Time periods are 2 weeks long and set 1 week apart, therefore they will overlap eachother by 1 week. This is done on purpose to ensure higher validity in the trends shown. 
# MAGIC
# MAGIC Movement metric which feeds into alerts and ordering of the insight tables is based on the average amount of prices returned by brokers for a quote in the last 2 weeks (P1), compared to the average of a 5 week period starting 1 week ago. However, both time periods aree offset by 3 days to allow for a maximum days to inception of 3 to be used.
# MAGIC
# MAGIC For example is the report is run on the 2nd of March then P1 will be February 14th to February 28th. The comparison period will be the 5 weeks leading up to the 21st of February.
