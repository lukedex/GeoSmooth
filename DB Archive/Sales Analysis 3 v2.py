# Databricks notebook source
!pip install polars

# COMMAND ----------

import pyarrow as pa
import polars as pl
spark_df = spark.sql("""select * from pricing.ds_cdl""")
df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))
#print(df)

# COMMAND ----------

import itertools
from datetime import datetime
 
total_quotes = len(df)
avg_conversion = df["Sold"].sum() / total_quotes

factors=["vehicleValue",
"annualMileage",
"mainDrivingExperienceMonths",
"ncdAllowed",
"ageDifference",
"FRTHVehicleGroup",
"ADVehicleGroup",
"WSRatingArea",
"PIVehicleGroup",
"marketingFactorsRatingArea",
"PDRatingArea",
"PDVehicleGroup",
"WSVehicleGroup",
"largeVehicleGroup",
"ADRatingArea",
"PIRatingArea",
"mainDriverAge",
"largeRatingArea",
"FRTHSmRatingArea",
"addDriverAge",
"mainDrivingExperience",
"yearsOwned",
"addDrivingExperience",
"ncdEarned",
"voluntaryExcess",
"brokerTenure",
"coveaRatingArea",
"HfxVehicleGroup",
"XSPIRatingArea",
"vehicleAge",
"vehicleAgeAtPurchase",
"mainOccupationType",
"tvRegion",
"mainEmploymentType",
"vehicleBodyType",
"mostSevereConviction",
"PDRCode",
"totFaultClaimsInLast5years",
"claims_split",
"latest_claim_split",
"vehicleKeeper",
"vehicleOwner",
"latestConviction",
"classOfUse",
"totConvictionsInLast5years",
"addOtherVehsOwned",
"addAccessToOtherVehs",
"accessToOtherVehs",
"ncdProtected",
"proposerHomeOwner",
"secondcarFlag",
"otherVehsOwned",
"cover"]
 
combinations = list(itertools.combinations(factors, 4))
print(len(combinations))

def create_group(feature_set):
    grouped = df.group_by(list(feature_set)).agg([pl.col("Sold").sum().suffix("_sum"),pl.col("Sold").count().alias("Quotes")])
    grouped = grouped.with_columns([pl.lit(grouped.columns[0]).alias("First_Factor"),
                     pl.lit(grouped.columns[1]).alias("Second_Factor"),
                     pl.lit(grouped.columns[2]).alias("Third_Factor"),
                     pl.lit(grouped.columns[3]).alias("Fourth_Factor")])
    grouped = grouped.rename({"Sold_sum" : "Sold",
                            feature_set[0] : "First_Factor_Level",
                            feature_set[1] : "Second_Factor_Level",
                            feature_set[2] : "Third_Factor_Level",
                            feature_set[3] : "Fourth_Factor_Level"})
    grouped = grouped.with_columns([
        (grouped["Quotes"] / total_quotes).alias("Exposure"),
        ((grouped["Sold"]/grouped["Quotes"])/avg_conversion).alias("Relative_Conversion")
        ]
        ).filter(pl.col("Exposure") >= 0.01)
    return grouped
#partition 0

print(df.shape)
print(total_quotes)
print(avg_conversion)


# COMMAND ----------

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

combinations_1,combinations_2,combinations_3=split(combinations,3)

# COMMAND ----------

# MAGIC %md
# MAGIC This cluster runs combinations_3

# COMMAND ----------

salesAnalysis = pl.DataFrame(schema = {
'First_Factor' : pl.datatypes.Utf8,
'Second_Factor' : pl.datatypes.Utf8,
'Third_Factor' : pl.datatypes.Utf8,
'Fourth_Factor' : pl.datatypes.Utf8,
'First_Factor_Level' : pl.datatypes.Utf8,
'Second_Factor_Level' : pl.datatypes.Utf8,
'Third_Factor_Level' : pl.datatypes.Utf8,
'Fourth_Factor_Level' : pl.datatypes.Utf8,
'Sold' : pl.datatypes.Int64,
'Quotes' : pl.datatypes.UInt32,
'Exposure' : pl.datatypes.Float64,
'Relative_Conversion' : pl.datatypes.Float64,
})

sets = []
  
st = datetime.now()
print(f"starting loop at {st}")
for count,i in enumerate(combinations_3):
    if not count%1000:
        print(count)
    sets.append(create_group(i))
 
print(f"finished groupby at {datetime.now()}")
print(f"process ran for {datetime.now() - st}")
partition_0 = pl.concat([salesAnalysis] + sets, how = 'diagonal')
print(partition_0.shape)
