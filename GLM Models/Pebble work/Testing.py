import statsmodels.api as sm
#import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import sklearn

Saga_NB_df = pd.read_csv(r"R:\Pricing & Actuarial\Pricing\Personal Lines Pricing - Motor\Technical\8. Optimisation\6. CDL\1. Jan22\2. Modelling\1. Cancellation\1. Data\CANCELLATION_NB_SAGA_Nov19-Nov21.csv")

X_numerical = Saga_NB_df[['Saga Delphi Score', 'Saga Factor Score', 'Saga_Cais_Score', 'Latest Fault Accident Claim', 'Latest Non Fault Accident Claim', 'Latest Windscreen Claim', 'Tot Fault Claims in Last 5 Years', 'Tot Fault Accident Claims', 'Tot Non Fault Accident Claims', 'Tot Windscreen Claims', 'Tot Fire Claims',	'Tot Theft Claims',	'Tot Vandalism Claims',	'Tot Other Claims',	'Tot Convictions',	'Latest Conviction', 'Annual Mileage',	'Main Driver Age',	'Main Driver Age Months', 'Main UK Residency', 'Main Driving Experience',	'Main Driving Experience Months', 	'Access to Other Vehs',	'Other Vehs Owned',	'Add Driver Age',	'Add Driving Experience',	'Age Difference','Add Access to Other Vehs', 'Add Other Vehs Owned', 'Vehicle Age', 'Years Owned', 'Vehicle Value', 'NCD Allowed', 'NCD Earned', 'Duration', 'Random Variable 5']]
y = Saga_NB_df['Cancelled']