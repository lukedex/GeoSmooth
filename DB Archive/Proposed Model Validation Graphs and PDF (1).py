# Databricks notebook source
!pip install scikit-learn
!pip install seaborn
!pip install nyoka
!pip install fpdf
!pip install shap

# COMMAND ----------

import shap
import warnings
warnings.filterwarnings("ignore")

# These are custom packages - I think we can import these using %run though??
#from useful_functions import *
#from feature_list_CPC import *

import numpy as np
import pandas as pd

from sklearn import preprocessing
#import missingno as msno

from sklearn import metrics
from sklearn.model_selection import train_test_split

from tqdm import tqdm #for a progress bar
import matplotlib.pyplot as plt

import datetime

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample

import seaborn as sns

import os # removes temp.png for pdf report

from nyoka import xgboost_to_pmml 
from IPython.display import clear_output
import statistics

# COMMAND ----------

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        if self.page_no() != 1:
            # Logo
            #self.image('covea-insurance.jpg', 10, 8, 33)
            # Arial bold 15
            self.set_font('Arial', 'B', 15)
            # Move to the right
            self.cell(80)
            # Title
            self.cell(30, 10, f'{model_name} Model Report', 'C')
            # Line break
            self.ln(30)

    # Page footer
    def footer(self):
        if self.page_no() != 1:
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
            
    def title_page(self, text, response, start, end):
        #self.image('covea-insurance.jpg', 10, 8, 33)
        self.ln(80)
        self.set_font('Arial', 'B', 24)
        self.multi_cell(0, 12, text, 0, align = 'C')
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, f'Report created on {str(datetime.date.today())}', 0, align = 'C')
        self.multi_cell(0, 5, f'Response: {response}', 0, align = 'C')
        self.multi_cell(0, 5, f'Data Range: {start} - {end}', 0, align = 'C')

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)
        
    def chapter_body(self, string):
        # Arial 10
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, string, 0, 1)
        
    def chapter_subtitle(self, string):
        # Arial 10
        self.ln(4)
        self.set_font('Arial', 'BU', 10)
        self.multi_cell(0, 5, string, 0, 1)
        
    def chapter_formula(self, string):
        # Arial 10
        self.ln(4)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, string, 0, align = 'C')
        self.ln(4)
        
    def appendix(self, string):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, string, 0, 1, 'L', 1)
        # Line break
        self.ln(4)

# COMMAND ----------

def shapley_feature_ranking(shap_values, X):
    feature_order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(np.abs(shap_values), axis=0)[i] for i in feature_order
            ][::-1],
        }
    )

# COMMAND ----------

def ICE_plot(X, model, feature, model_name, centered = False, show = True):
    num = X.shape[0]
    
    min_val = np.floor(np.min(X[feature]))
    max_val = np.ceil(np.max(X[feature]))
    N = int(min(max_val - min_val,100))
    
    values = np.linspace(min_val, max_val, N + 1)
    title = f'ICE Plot for {feature}'
    ylabel = model_name
    xlabel = feature
    
    ICE_values = []
    pdp_values = []
    
    for value in values:
        X_pdp = X.copy() #Is this a mistake? should this be 'X'
        X_pdp[feature] = value
        mod_vals = model.predict(X_pdp)
        ICE_values.append(mod_vals)
        pdp_values.append(np.mean(mod_vals))
    
    if centered:
        cICE_lst = []

        for ICE in list(zip(*ICE_values)):
            ICE2 = np.array(ICE)
            cICE = ICE2 / ICE2[0]
            cICE_lst.append(cICE)

        ICE_values = list(zip(*cICE_lst))
        pdp_values = np.array(pdp_values) / np.array(pdp_values)[0]
        
        title = f'cICE Plot for {feature}'
        ylabel = f'Indexed {model_name}'
        
    fig, ax = plt.subplots(figsize=(12,8))
    ax.grid()
    ax.plot(values, ICE_values, alpha = 0.2, color = 'cornflowerblue')
    ax.plot(values, pdp_values, color = 'red', linewidth=3)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
    if show:
        plt.show()
    else:
        return plt.gcf()
    

# COMMAND ----------

"""
Things to make:
X200 and X5000

[Features] for checking.
"""
import mlflow
logged_model = 'runs:/8484dff47dee4d0a884a1b5ccc23eed0/model'

# Load model as a PyFuncModel. - Can we extract model from this?
#loaded_model = mlflow.pyfunc.load_model(logged_model)

#This line imports the sklearn pipeline itself - much easier to use!
loaded_model = mlflow.sklearn.load_model(logged_model)
# Predict on a Pandas DataFrame.
#import pandas as pd
#loaded_model.predict(pd.DataFrame(data))


import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="1bf5f6c8fcde4a4e84fb195cafed7756", artifact_path="data", dst_path=input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

X200=df_loaded.drop(columns=['avgprice15','_automl_split_col_0000'], inplace=False).sample(200)
X5000=df_loaded.drop(columns=['avgprice15','_automl_split_col_0000'], inplace=False).sample(5000)
test_loaded=df_loaded[df_loaded['_automl_split_col_0000']=='test']

# COMMAND ----------

# MAGIC %md
# MAGIC #Scoring up all of August test data for analysis
# MAGIC - Further work: Could look at doing this for September as well? ensure it's working well - maybe in another notebook?

# COMMAND ----------

analysis=pd.DataFrame()
analysis['avgprice15']=test_loaded['avgprice15']
analysis['prediction']=loaded_model.predict(test_loaded)

# COMMAND ----------

# MAGIC %md
# MAGIC #Distribution of Predictions vs Actuals

# COMMAND ----------

dor_df=pd.DataFrame(analysis['prediction'], columns=['prediction'])
dor_df['avgprice15']=test_loaded['avgprice15']
dor_df=dor_df[dor_df['prediction']>0] #found 1 example of a negative market prediction
dor_df.sort_values('avgprice15',inplace=True)
plot_df=dor_df[(dor_df['avgprice15']<5000)&(dor_df['prediction']<5000)]
plot_df.plot.hist(bins=1000, alpha=0.5)

# COMMAND ----------

factor='mainDriverAge'
dor_df=dor_df[['prediction','avgprice15']] #Removes previous factor
dor_df[factor]=test_loaded[factor]
dor_df.sort_values(factor,inplace=True)

dor_df_filter=dor_df[dor_df[factor]>0]
dor_df_filter.groupby(by=factor).mean().plot(alpha=0.5)

#dor_df_filter.groupby(by=factor).value_counts().plot(kind='bar')

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

factor_list=['mainDriverAge','ncdallowed_capped','mainDrivingExperience','transUnionConceptsJF','quote_lag_calc','maindriverexperience_uk','perc_drv_uk','yearsOwned','vehicleAgeAtPurchase','mainmaritalstatus','mainLicenceType']


for factor in factor_list:

    dor_df=dor_df[['prediction','avgprice15']] #Removes previous factor
    dor_df[factor]=test_loaded[factor]
    dor_df.sort_values(factor,inplace=True)
    if factor == 'transUnionConceptsJF':
        dor_df_filter=dor_df[dor_df[factor]>0]
    else:
        dor_df_filter=dor_df

    trace1 = go.Bar(
        x=dor_df_filter.groupby(by=factor, as_index=False).count()[factor],
        y=dor_df_filter.groupby(by=factor, as_index=False).count()['avgprice15'],
        name=factor,
        marker=dict(color='white')
        )

    trace2 = go.Scatter(
        x=dor_df_filter.groupby(by=factor, as_index=False).count()[factor],
        y=dor_df_filter.groupby(by=factor).mean()['prediction'],
        name='Prediction',
        yaxis='y2',
        marker=dict(color='black')
        )

    trace3 = go.Scatter(
        x=dor_df_filter.groupby(by=factor, as_index=False).count()[factor],
        y=dor_df_filter.groupby(by=factor).mean()['avgprice15'],
        name='Average',
        yaxis='y2',
        marker=dict(color='blue')
        )


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace1)
    fig.add_trace(trace2,secondary_y=True)
    fig.add_trace(trace3,secondary_y=True)

    fig['layout'].update( title = factor,xaxis=dict(
        ))
    display(fig)


# COMMAND ----------

analysis['prediction_floor']=np.where(analysis['prediction']<85.43, 85.43,analysis['prediction'])
analysis['prediction_floor']

# COMMAND ----------

#feature_names

# COMMAND ----------

# MAGIC %md
# MAGIC # Residual Distribution

# COMMAND ----------

# Looking at residuals
residual_df=pd.DataFrame()
residual_df['residuals']=plot_df['avgprice15']-plot_df['prediction']
residual_df.plot.hist(bins=100)

#0.2% of residuals out by 2000

#Mean residual is 105.6
#abs(residual_df).mean() 



# COMMAND ----------

#loaded_model['preprocessor'].get_feature_names_out()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance
# MAGIC  - Use this cell to output feature importance by either 'split' or 'gain' 
# MAGIC  - Output can be a plot or dataframe

# COMMAND ----------

"""This cell is used to display(new_df) of the fulldataset to then download via this browser. Or to output feature importance graphs. 
You need to uncomment certain lines to get this working for what you need."""

fi_df=pd.DataFrame()
fi_df['Feature_Name']=loaded_model["preprocessor"].get_feature_names_out()
fi_df['Feature_Gain_Values']=loaded_model['regressor'].booster_.feature_importance(importance_type='gain') #Can change to 'split' for number of splits
#fi_df.set_index('Feature_Name', inplace=True)
fi_df.sort_values('Feature_Gain_Values', ascending=False, inplace=True)
#fi_df.tail(20).plot(kind='barh')
display(fi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Working on getting Charlies PDF creation script for interpretability working in Databricks on LGBM

# COMMAND ----------

model_name='Market Model OCT23'
response='AvgPrice15'
start='1st August 2023'
end='31st August 2023'
feature_names=loaded_model['preprocessor'].get_feature_names_out()
#X5000 = #defined already
#X200 = #defined already
#Transformed data with new columns names
X200_df = pd.DataFrame(data=loaded_model['preprocessor'].transform(X200),columns=feature_names)           



# perils includes the chosen trial to obtain hyperparmeters.

""" I think this is just loading a model?
peril = peril_tup[0]
chosen_trial = peril_tup[1]
response = peril_tup[2]
weight = peril_tup[3]
freq_sev_prop = peril.split("_",1)[1]  
start = df['Exposure_Start'].min()
end = df['Exposure_End'].max()

if freq_sev_prop == 'Sev':
    scalar = 1
else:
    scalar = 100


model = xgb.Booster()
model.load_model(f'JSON_Files/xgboost_{peril}.json')
"""


####
#### TITLE PAGE ####
####

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.set_margins(15,10)
pdf.title_page(f'{model_name} Model Report', response, start, end)

pdf.add_page()

pdf.chapter_title(1, "Hyperparameters")

#trial_df = pd.read_csv(f'Hyperparameters/{peril}_HPs_forPMML.csv')
#chosen_trial_df = trial_df[trial_df['trial_num'] == chosen_trial]
#print(i,  loaded_model['regressor'].get_params()[i])

for col in loaded_model['regressor'].get_params().keys():
    pdf.chapter_body(f"{col} : {loaded_model['regressor'].get_params()[col]}")



################################    
#### FEATURE List ############## ##
################################ 

pdf.add_page()    
pdf.chapter_title(2, "Features Used")
    
for factor in feature_names:
    pdf.chapter_body(f"{factor}")

################################    
#### FEATURE IMPORTANCE ######## ##COMPLETE
################################    

pdf.add_page()    
pdf.chapter_title(2, "Feature Importance")
    
fig, ax = plt.subplots(figsize=(7,15))
temp_df=pd.DataFrame()
temp_df['Feature_Name']=loaded_model["preprocessor"].get_feature_names_out()
temp_df['Feature_Gain_Values']=loaded_model['regressor'].booster_.feature_importance(importance_type='gain') #Can change to 'split' for number of splits
temp_df.set_index('Feature_Name', inplace=True)
temp_df.sort_values('Feature_Gain_Values', ascending=True, inplace=True)
temp_df.tail(30).plot(kind='barh',  ax=ax)
plt.savefig('temp.png', bbox_inches="tight")
plt.close()
scale = 14
pdf.image('temp.png', w = 12*scale, h = 14*scale)
os.remove('temp.png')



explainer = shap.TreeExplainer(loaded_model['regressor'])
shap_values = explainer.shap_values(loaded_model['preprocessor'].transform(X5000)) 
shap_values2 = explainer(loaded_model['preprocessor'].transform(X5000)) 
shap_values2.feature_names=feature_names

#########################
#### BEES ARRRGGGHHH #### ##COMPLETE
#########################

pdf.add_page()  
pdf.chapter_title(3, "Beeswarm Plot")
fig, ax1 = plt.subplots()
ax1.grid()
shap.summary_plot(shap_values, features=feature_names, max_display = 20,show=False, plot_size = (12,8))
plt.savefig('temp1.png', bbox_inches="tight")
plt.close()
scale = 14
pdf.image('temp1.png', w = 12*scale, h = 8*scale)
os.remove('temp1.png')
pdf.chapter_body('Features in the above plot are ordered by the mean absolute SHAP value.')
pdf.add_page() 
fig, ax1 = plt.subplots()
ax1.grid()
shap.plots.beeswarm(shap_values=shap_values2, order=shap_values2.abs.max(0),show=False,max_display = 20,plot_size = (12,8))
plt.savefig('temp1a.png', bbox_inches="tight")
plt.close()
scale = 14
pdf.image('temp1a.png', w = 12*scale, h = 8*scale)
os.remove('temp1a.png')
pdf.chapter_body('Features in the above plot are ordered by the max absolute SHAP value. This helps identify any extreme cases.')

##########################
#### DEPENDENCE PLOTS ####
##########################

pdf.add_page()  
pdf.chapter_title(4, "Dependence Plots")

analysis_list = set(fi_df['Feature_Name'].head(40))# + usual_suspects)

interact_with = None                        # can set to 'auto'
for feature in analysis_list:
    if feature.find('trans') != -1:
        fig, ax1 = plt.subplots()
        ax1.grid()
        shap.dependence_plot(feature, shap_values, features=loaded_model['preprocessor'].transform(X5000), feature_names=feature_names, interaction_index=interact_with, ax = ax1, title = f'Dependence Plot for {feature}', alpha = 0.5,show=False, xmin=-1)
        plt.savefig(f'temp_{feature}.png', bbox_inches="tight")
        plt.close()
        scale = 14
        pdf.image(f'temp_{feature}.png', w = 12*scale, h = 8*scale)
        os.remove(f'temp_{feature}.png')
    else:
        fig, ax1 = plt.subplots()
        ax1.grid()
        shap.dependence_plot(feature, shap_values, features=loaded_model['preprocessor'].transform(X5000), feature_names=feature_names, interaction_index=interact_with, ax = ax1, title = f'Dependence Plot for {feature}', alpha = 0.5,show=False)
        plt.savefig(f'temp_{feature}.png', bbox_inches="tight")
        plt.close()
        scale = 14
        pdf.image(f'temp_{feature}.png', w = 12*scale, h = 8*scale)
        os.remove(f'temp_{feature}.png')



    """ 
    
    ##NEED TO ADD IN RULES FOR TRANSUNION FACTORS AS THE HAVE -99997 WHICH IS SKEWING CHARTS MASSIVELY. CAN'T FILTER FOR OTHER GRAPHS AS THEY INTERACT WITH OTHER FEATURES
    
    
    fig, ax2 = plt.subplots()
    ax2.grid()
    shap.dependence_plot(feature, shap_values, features=loaded_model['preprocessor'].transform(X5000), feature_names=feature_names, interaction_index=interact_with, ax = ax2, title = f'Dependence Plot for {feature} within 5th to 9th percentile', alpha = 0.5,show=False, xmin="percentile(10)",xmax="percentile(100)")
    plt.savefig(f'temp_{feature}.png', bbox_inches="tight")
    plt.close()
    scale = 14
    pdf.image(f'temp_{feature}.png', w = 12*scale, h = 8*scale)
    os.remove(f'temp_{feature}.png')
"""

###########################
#### INTERACTION PLOTS ####
###########################

pdf.add_page()  
pdf.chapter_title(5, "Interactions")
""" NEEDS WORK
peril2 = peril.replace("_", "")

vehicle_group_list = [('Main_Driver_Age', f'{peril2}VehicleGroup2022'), (f'{peril2}VehicleGroup2022', 'Access_to_Other_Vehs'), (f'{peril2}VehicleGroup2022', 'Vehicle_Age')]
rating_area_list = [(f'{peril}_Rating_Area_2022', 'Garaged')]

loop_list = top20features + vehicle_group_list + rating_area_list

for interaction in loop_list:
    feature = interaction[0]
    interact_with = interaction[1]                      # can set to 'auto'
    fig, ax1 = plt.subplots()
    ax1.grid()
    try:
        shap.dependence_plot(feature, shap_values, X5000_display, interaction_index=interact_with, ax = ax1, title = f'Interaction between {feature} and {interact_with}', alpha = 0.5,show=False)
    except:
        continue
    plt.savefig(f'temp_int_{feature}{interact_with}.png', bbox_inches="tight")
    plt.close()
    scale = 14
    pdf.image(f'temp_int_{feature}{interact_with}.png', w = 12*scale, h = 8*scale)
    os.remove(f'temp_int_{feature}{interact_with}.png')

"""
###################
#### ICE PLOTS ####
###################

pdf.add_page()  
pdf.chapter_title(6, "Individual Conditional Expectation (ICE) Plots")
for feature in X200_df.columns:
    # Regular ICE Plot
    ICE_plot(X200_df, loaded_model['regressor'], feature, model_name=model_name, centered = False, show = False)
    plt.savefig(f'ICE_{feature}.png', bbox_inches="tight")
    plt.close()
    scale = 14
    pdf.image(f'ICE_{feature}.png', w = 12*scale, h = 8*scale)
    os.remove(f'ICE_{feature}.png')
    
    # Centered ICE Plot
    ICE_plot(X200_df, loaded_model['regressor'], feature, model_name=model_name, centered = True, show = False)
    plt.savefig(f'cICE_{feature}.png', bbox_inches="tight")
    plt.close()
    scale = 14
    pdf.image(f'cICE_{feature}.png', w = 12*scale, h = 8*scale)
    os.remove(f'cICE_{feature}.png')

####
#### WATERFALL PLOTS ####
####
###
##
#

pdf.add_page()
pdf.chapter_title(7, 'Extreme Value Checking - Waterfall Plots')
""" REMOVED FOR NOW
f = lambda x: model.inplace_predict(x)*scalar # THIS IS A REALLY HACKY WAY ROUND THE ROUNDING ISSUE
explainer = shap.Explainer(f, X5000)
shap_values = explainer(X5000)

modval_list = model.inplace_predict(X5000)
median = statistics.median(modval_list)
big_list = []
small_list = []

for i in range(5):
    big = np.argmax(modval_list)
    small = np.argmin(modval_list)

    big_list.append(big)
    small_list.append(small)

    # Move max and min to middle to find next biggest or smallest
    modval_list[big] = median
    modval_list[small] = median

# could just put this in previous loop. But want to group bigs and smalls together.
for ind in big_list:
    fig, ax1 = plt.subplots()
    shap.plots.waterfall(shap_values[ind], max_display = 20, show = False)
    plt.savefig(f'waterfall_{ind}.png', bbox_inches="tight")
    plt.close()
    scale = 14
    pdf.image(f'waterfall_{ind}.png', w = 12*scale, h = 12*scale)
    os.remove(f'waterfall_{ind}.png')
for ind in small_list:
    fig, ax1 = plt.subplots()
    shap.plots.waterfall(shap_values[ind], max_display = 20, show = False)
    plt.savefig(f'waterfall_{ind}.png', bbox_inches="tight")
    plt.close()
    scale = 14
    pdf.image(f'waterfall_{ind}.png', w = 12*scale, h = 12*scale)
    os.remove(f'waterfall_{ind}.png')
"""
####
#### APPENDIX EXPLAINER ####
####

pdf.add_page()
pdf.appendix('Appendix - Interpreting these Plots')

pdf.chapter_subtitle('Document Author')
pdf.chapter_body('This report has been created by Luke Dexter.')

pdf.chapter_subtitle('Documentation Credit')
pdf.chapter_body('The original reporting structure of this document was designed by Charlie Sinclair and has been re-used for this model report. All credit goes to Charlie for the huge effort required to create a document like this.')

pdf.chapter_subtitle('Shapely Values (SHAP values)')
pdf.chapter_body('A lot of the plots rely on SHAP values. The simple interpretation of the SHAP value of a feature is "How much is that particular feature contributing to the overall prediction compared to the mean?".')
pdf.chapter_body('For example, suppose the average PD Frequency of a dataset is 0.03 and that a particular policy has a prediction of 0.045. If that policy has a Main Driver Age of 18, which has a SHAP value of 0.01, then this is saying that 0.01 of the 0.015 increase can be attributed to the Main Driver Age factor.')
pdf.chapter_body('For more detail on how this is calculated please read Chapter 9.5 of the following e-book: https://christophm.github.io/interpretable-ml-book/')

pdf.chapter_subtitle('Chapter 1')
pdf.chapter_body('This section simply states the hyperparameters used to train the GBM.')

pdf.chapter_subtitle('Chapter 2')
pdf.chapter_body('This is a feature importance plot. It ranks the different features based on how important the F-Score metric perceives that feature to be.')
pdf.chapter_formula('F-Score = The Number of Times that Feature is split on in the GBM')

pdf.chapter_subtitle('Chapter 3')
pdf.chapter_body('This chapter contains two beeswarm plots. These are similar to the feature importance plot but also show how a particular feature\'s SHAP values are distributed.')
pdf.chapter_body('In the first beeswarm plot the features are ordered by the mean absolute SHAP value of that particular feature. This order places more emphasis on broad average impact, and less on rare but high magnitude impacts.')
pdf.chapter_body('The second beeswarm plot orders the features by the maximum absolute SHAP value. This aims to find features with high impacts for individual people.')

pdf.chapter_subtitle('Chapters 4 & 5')
pdf.chapter_body('These chapters display a number of dependence plots for various features. A dependence plot plots, for all policies in the dataset, the SHAP value of a particular feature against the value of the feature.')
pdf.chapter_body('Vertical dispersion at a single value of a particular feature represents interaction effects with other features. To try to find these interaction effects the SHAP values are coloured based on the value of another feature.')

pdf.chapter_subtitle('Chapter 6')
pdf.chapter_body('This chapter shows ICE and PDP plots. Each line in an ICE plot represents a different policy. The line is calculated by generating a prediction for each value of that factor.')
pdf.chapter_body('The PDP is the bold line and represents the average of all the ICE plots.')
pdf.chapter_body('ICE plots that are all congruous indicates there are minimal interactions with that particular factor.')
pdf.chapter_body('The second plot of the page contains centered ICE plots. This indexes the value at the lowest feature value for ease of reading.')

pdf.chapter_subtitle('Chapter 7')
pdf.chapter_body('This chapter has waterfall plots. Waterfall plots are designed to display explanations for individual predictions. The bottom of a waterfall plot starts as the expected value of the model output, and then each row shows how the positive (red) or negative (blue) contribution of each feature moves the value from the expected model output over the background dataset to the model output for this prediction.')

pdf.output(f'/dbfs/FileStore/MMOCT23_Outputs/Market_Model_OCT23_report.pdf', 'F')
