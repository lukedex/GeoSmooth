# Databricks notebook source
# MAGIC %md
# MAGIC # VERY SLOW!! DO NOT USE FOR LARGE AMOUNTS OF DATA..
# MAGIC Takes a pandas dataframe with 2 columns 'lat' and 'long' and adds on a column to this dataset called 'postcode'. \
# MAGIC Note: This package is very useful, it can give you lots of information about a long/lat or postcode etc so has further applications beyond this.

# COMMAND ----------

### Import your data 

df=spark.sql("""select *from pricing.enrich_crime_by_lat_longs_jb 
             """)             
df=df.toPandas()

## Create 2 new fields, if your columns aren't already named correctly
df['lat']=df['Latitude']
df['long']=df['Longitude']

postcode_df=spark.sql("""select *from pricing.ukpostcodes_csv
             """)         
postcode_df=postcode_df.toPandas()

# COMMAND ----------

from math import radians, cos, sin, asin, sqrt
def dist(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    distance=sqrt(dlat**2 + dlon**2)
    return distance

def find_nearest(lat, long):
    #Added this in. It cuts down the dataset to compare to.. saves comparing each round to the full postcode file (I think this should work with very high accuracy.. maybe it can fail but to 2 decimal places you still get 1000 matches)
    temp_df=postcode_df[(postcode_df['latitude'].round(1)==lat.round(1)) & (postcode_df['longitude'].round(1)==long.round(1))]
    #Compares each of the above split, brings back closest postcode.
    distances = temp_df.apply(
        lambda row: dist(lat, long, row['latitude'], row['longitude']), 
        axis=1)
    return temp_df.loc[distances.idxmin(), 'postcode']

df_10['postcode'] = df_10.apply(
    lambda row: find_nearest(row['lat'], row['long']), 
    axis=1)

# To check the data frame if it has a new column of postcode name (for each and every crime's location in the list)
df_10.head()

# COMMAND ----------

2**3

# COMMAND ----------

df_10
