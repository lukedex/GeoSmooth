# Databricks notebook source
# MAGIC %md
# MAGIC ###Notes:
# MAGIC - The first 2 cells here are ones which do not need changing. Simply run these 2 cells.
# MAGIC - The third cell needs input 
# MAGIC   - Name of the SQL table, including 'pricing.' if within the Pricing schema
# MAGIC   - List of factors to ignore
# MAGIC   - Column name for Actuals
# MAGIC   - Column name for Expected/Predictions
# MAGIC - Forth cell also does not need editing. Running it will perform calculations to find the factors which have moved the most.
# MAGIC

# COMMAND ----------

import pandas as pd
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px

# COMMAND ----------

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert(len(actual) == len(pred))
    expo = np.ones(len(actual))
    all1 = np.asarray(np.c_[ actual, pred, np.linspace(0, 1, len(actual)),expo], dtype=float)
    all = all1[ np.lexsort((all1[:,2], -1*all1[:,1])) ]
    actTotal = all[:,0].sum()
    expoTotal = all1[:,3].sum()
    actCumSum = all[:,0].cumsum()
    expoCumSum = all1[:,3].cumsum()
    g = (actCumSum*all1[:,3]).sum()/(actTotal*expoTotal)
    return 2*(g - 0.5)

def model_drift_analysis(sql_dataset=None, 
                         pandas_dataframe=None,
                         actuals_column=None,
                         predictions_column=None, 
                         exposure_column=None,
                         columns_to_ignore=None):
    
    """
    Inputs:
    sql_dataset : string 
        Name of the sql table to be used, example 'pricing.table_for_drift_report'
    pandas_dataframe : Pandas DataFrame
        Name of the pandas dataframe
    actuals_column : string
        Name of the column within the pandas dataframe (or SQL table) which holds the 'Actuals' for drift report.
    predictions_column : string
        Name of the column within the pandas dataframe (or SQL table) which holds the 'Predictions' for drift report.
    exposure_column : sting
        Name of the column within the pandas dataframe (or SQL table) which holds the 'Predictions' for drift report.
    columns_to_ignore: list
        List of columns to ignore, such as primary keys etc. Example: ['UUID','Hub_Timestamp','Quote_Date','QuoteID']
    

    Returns:
    Time agnostic analysis : Pandas Dataframe 
    Analysis split by analysis week : Pandas Dataframe
    
    """
    #Gives the option for a SQL table to be called rather than a PandasDF.
    if sql_dataset!=None:
        print(f'Ingesting SQL table {sql_dataset}')
        spark_df_=spark.sql(f"select * from {sql_dataset}")
        df_=spark_df_.toPandas()
        print('SQL Dataset ingested.')
    else:
        df_=pandas_dataframe


    #Remove the columns not needed for drift report
    if isinstance(columns_to_ignore, list):
        factors_to_monitor = [x for x in df_.columns if x not in columns_to_ignore]
        print(f'{columns_to_ignore} sucessfully removed from analysis list.')
    else:
        print("columns_to_ignore is not of type 'list'. All columns will be checked.")
        factors_to_monitor = df_.columns

    #Create a blank dataframe to be populated.
    analysis_df_=pd.DataFrame(columns=['Factor','Level','Count','Actuals','Predicted','Gini_Norm','MAE',])
   
    factor_list=[]
    level_list=[]
    count_list=[]
    actuals_list=[]
    predicted_list=[]
    gini_list=[]
    mae_list=[]

    print('Creating overall statistics.')
    #Manually Add in some Overall Statistics
    factor_list.append('Overall')
    level_list.append('Overall')
    count_list.append(len(df_))
    actuals_list.append(df_[actuals_column].mean())
    predicted_list.append(df_[predictions_column].mean())
    gini_list.append(gini(df_[actuals_column],
                            df_[predictions_column])/gini(df_[actuals_column],
                            df_[actuals_column]))
    mae_list.append(mean_absolute_error(df_[actuals_column],
                                        df_[predictions_column]))

    print('Creating feature level statistics.')
    #Loop thought factors and record the outputs per level of each factor
    for factor in factors_to_monitor:
        print(f'{factor} analysis beginning. {df_[factor].nunique()} levels found.')
    #find unique values of that factor
        levels_list=df_[factor][df_[factor].isnull()==False].unique()
        for level in levels_list:
            factor_list.append(factor)
            level_list.append(level)
            count_list.append(len(df_[factor][df_[factor]==level]))
            actuals_list.append(df_[actuals_column][df_[factor]==level].mean())
            predicted_list.append(df_[predictions_column][df_[factor]==level].mean())
            #Calculate Gini and MAE of each level (Needs to be done carefully.. it can be extremely volatile and mis-leading)
            #If there are pots with low levels of exposure
            gini_list.append(gini(df_[actuals_column][df_[factor]==level],
                                df_[predictions_column][df_[factor]==level])/gini(df_[actuals_column][df_[factor]==level],
                                df_[actuals_column][df_[factor]==level]))
            mae_list.append(mean_absolute_error(df_[actuals_column][df_[factor]==level],
                                                df_[predictions_column][df_[factor]==level]))
            

    analysis_df_['Factor']=factor_list
    analysis_df_['Level']=level_list
    analysis_df_['Count']=count_list
    analysis_df_['Actuals']=actuals_list
    analysis_df_['Predicted']=predicted_list
    analysis_df_['Gini_Norm']=gini_list
    analysis_df_['MAE']=mae_list
    print('All factors completed.')


   #Create a blank dataframe to be populated.
    time_analysis_df_=pd.DataFrame(columns=['Factor','Level','Count','Analysis_Week','Actuals','Predicted','Gini_Norm','MAE',])
    print('Starting time analysis.')
    time_factor_list=[]
    time_level_list=[]
    time_count_list=[]
    time_analysis_week_list=[]
    time_actuals_list=[]
    time_predicted_list=[]
    time_gini_list=[]
    time_mae_list=[]

    for week in df_['analysis_week'].unique():
        print(f'Starting analysis_week = {week} loop.')
        time_df_ = df_[df_['analysis_week']==week]

        print(f'Creating overall statistics for {week}')
        #Manually Add in some Overall Statistics
        time_factor_list.append('Overall')
        time_level_list.append('Overall')
        time_count_list.append(len(time_df_))
        time_analysis_week_list.append(week)
        time_actuals_list.append(time_df_[actuals_column].mean())
        time_predicted_list.append(time_df_[predictions_column].mean())
        time_gini_list.append(gini(time_df_[actuals_column],
                                time_df_[predictions_column])/gini(time_df_[actuals_column],
                                time_df_[actuals_column]))
        time_mae_list.append(mean_absolute_error(time_df_[actuals_column],
                                            time_df_[predictions_column]))

        print('Creating feature level statistics.')
        #Loop thought factors and record the outputs per level of each factor
        for factor in factors_to_monitor:
            print(f'{factor} analysis beginning. {time_df_[factor].nunique()} levels found.')
        #find unique values of that factor
            levels_list=time_df_[factor][time_df_[factor].isnull()==False].unique()
            for level in levels_list:
                time_factor_list.append(factor)
                time_level_list.append(level)
                time_count_list.append(len(time_df_[factor][time_df_[factor]==level]))
                time_analysis_week_list.append(week)
                time_actuals_list.append(time_df_[actuals_column][time_df_[factor]==level].mean())
                time_predicted_list.append(time_df_[predictions_column][time_df_[factor]==level].mean())
                #Calculate Gini and MAE of each level (Needs to be done carefully.. it can be extremely volatile and mis-leading)
                #If there are pots with low levels of exposure
                time_gini_list.append(gini(time_df_[actuals_column][time_df_[factor]==level],
                                    time_df_[predictions_column][time_df_[factor]==level])/gini(time_df_[actuals_column][time_df_[factor]==level],
                                    time_df_[actuals_column][time_df_[factor]==level]))
                time_mae_list.append(mean_absolute_error(time_df_[actuals_column][time_df_[factor]==level],
                                                    time_df_[predictions_column][time_df_[factor]==level]))
                

    time_analysis_df_['Factor']=time_factor_list
    time_analysis_df_['Level']=time_level_list
    time_analysis_df_['Count']=time_count_list
    time_analysis_df_['Analysis_Week']=time_analysis_week_list
    time_analysis_df_['Actuals']=time_actuals_list
    time_analysis_df_['Predicted']=time_predicted_list
    time_analysis_df_['Gini_Norm']=time_gini_list
    time_analysis_df_['MAE']=time_mae_list
    print('Time analysis of factors completed.')

    return analysis_df_ , time_analysis_df_

def make_plotly_graph(dataframe,
                      factor_name):

    sorted_results=dataframe.sort_values(by=['Factor','Level'])    
    overall_gini = sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'].values
    overall_mae = sorted_results['MAE'][sorted_results['Factor']=='Overall'].values
    sorted_results['AvE'] =  sorted_results['Actuals']/sorted_results['Predicted']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=sorted_results['Level'][sorted_results['Factor']==factor_name], 
                y=sorted_results['Gini_Norm'][sorted_results['Factor']==factor_name]/overall_gini, 
                name='Gini Norm Score',
                marker = {'color' : 'red'}
                ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=sorted_results['Level'][sorted_results['Factor']==factor_name], 
                y=sorted_results['MAE'][sorted_results['Factor']==factor_name]/overall_mae, 
                name='MAE',
                marker = {'color' : 'blue'}
                ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=sorted_results['Level'][sorted_results['Factor']==factor_name], 
                y=sorted_results['AvE'][sorted_results['Factor']==factor_name],
                name='AvE',
                marker = {'color' : 'green'}
                ),
        secondary_y=True
    )

    fig.add_trace(
        go.Bar(x=sorted_results['Level'][sorted_results['Factor']==factor_name], 
                y=sorted_results['Count'][sorted_results['Factor']==factor_name],  
                name="Quote Volume",
                marker = {'color' : 'yellow'}),
    )

    # Add figure title
    fig.update_layout(
        title_text="Relative Gini_Norm, MAE, AvE scores per level."
    )

    # Set x-axis title
    fig.update_xaxes(title_text=f"{factor_name}")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Count</b> of quotes", secondary_y=False)
    fig.update_yaxes(title_text="<b>Relativities</b> of metrics to global.", secondary_y=True)
    return fig


    # I want a colour scale imbedded here for the colours to make it easier to track.

def make_gini_plotly_graph(dataframe,
                      factor_name):

    sorted_results=dataframe.sort_values(by=['Factor','Analysis_Week','Level'])    
    overall_gini = sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'].values
    overall_mae = sorted_results['MAE'][sorted_results['Factor']=='Overall'].values
    sorted_results['AvE'] =  sorted_results['Actuals']/sorted_results['Predicted']

    fig = px.line(sorted_results[sorted_results['Factor']==factor_name], 
                  x='Level', 
                  y='Gini_Norm', 
                  color='Analysis_Week',
                  color_discrete_sequence=["#FF0000","#CC0033","#990066","#660099","#3300CC","#0000FF"])
    
    # Add figure title
    fig.update_layout(
        title_text="Relative Gini_Norm split by level over time."
    )

    # Set x-axis title
    fig.update_xaxes(title_text=f"{factor_name}")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Gini Norm</b> per week", secondary_y=False)
    return fig

    # I want a colour scale imbedded here for the colours to make it easier to track.

def make_ave_plotly_graph(dataframe,
                      factor_name):

    sorted_results=dataframe.sort_values(by=['Factor','Analysis_Week','Level'])    
    overall_gini = sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'].values
    overall_mae = sorted_results['MAE'][sorted_results['Factor']=='Overall'].values
    sorted_results['AvE'] =  sorted_results['Actuals']/sorted_results['Predicted']

    fig = px.line(sorted_results[sorted_results['Factor']==factor_name], 
                  x='Level', 
                  y='AvE', 
                  color='Analysis_Week',
                  color_discrete_sequence=["#FF0000","#CC0033","#990066","#660099","#3300CC","#0000FF"])
    
    # Add figure title
    fig.update_layout(
        title_text="Actual vs Expected split by level over time."
    )

    # Set x-axis title
    fig.update_xaxes(title_text=f"{factor_name}")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Actuals / Expected</b>", secondary_y=False)
    return fig

    # I want a colour scale imbedded here for the colours to make it easier to track.

def make_stacked_count_plotly_graph(dataframe,
                      factor_name):

    sorted_results=dataframe.sort_values(by=['Factor','Analysis_Week','Level'])    
    overall_gini = sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'].values
    overall_mae = sorted_results['MAE'][sorted_results['Factor']=='Overall'].values
    sorted_results['AvE'] =  sorted_results['Actuals']/sorted_results['Predicted']

    fig = px.histogram(sorted_results[sorted_results['Factor']==factor_name], 
                  x='Level', 
                  y='Count', 
                  color='Analysis_Week',
                  barnorm='percent',
                  color_discrete_sequence=["#FF0000","#CC0033","#990066","#660099","#3300CC","#0000FF"])
    
    # Add figure title
    fig.update_layout(
        title_text=f"Exposure trend of {factor_name }over time."
    )

    # Set x-axis title
    fig.update_xaxes(title_text=f"Analysis Week")


    # Set y-axes titles
    fig.update_yaxes(title_text="<b>% Split of counts</b> per week", secondary_y=False)
    return fig


    # I want a colour scale imbedded here for the colours to make it easier to track.

def make_count_plotly_graph(dataframe,

                      factor_name):

    sorted_results=dataframe.sort_values(by=['Factor','Analysis_Week','Level'])    
    overall_gini = sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'].values
    overall_mae = sorted_results['MAE'][sorted_results['Factor']=='Overall'].values
    sorted_results['AvE'] =  sorted_results['Actuals']/sorted_results['Predicted']

    fig = px.histogram(sorted_results[sorted_results['Factor']==factor_name], 
                  x='Level', 
                  y='Count', 
                  color='Analysis_Week',
                  color_discrete_sequence=["#FF0000","#CC0033","#990066","#660099","#3300CC","#0000FF"]
                  )
    
    # Add figure title
    fig.update_layout(
        title_text=f"Exposure trend of {factor_name} over time."
    )

    # Set x-axis title
    fig.update_xaxes(title_text=f"Analysis Week")

    #fig.update_layout(barmode='relative')

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>% Split of counts</b> per week", secondary_y=False)
    return fig

def make_factor_summary_plotly_graph(dataframe, factor_name):

    ave_fig=make_ave_plotly_graph(dataframe, factor_name)
    gini_fig=make_gini_plotly_graph(dataframe, factor_name)
    exp_fig=make_count_plotly_graph(dataframe, factor_name)
    exp_stack_fig=make_stacked_count_plotly_graph(dataframe, factor_name)
    

    ave_traces = []
    gini_traces = []
    exp_traces = []
    exp_stack_traces = []

    for trace in range(len(ave_fig["data"])):
        ave_traces.append(ave_fig["data"][trace])
    for trace in range(len(gini_fig["data"])):
        gini_traces.append(gini_fig["data"][trace])
    for trace in range(len(exp_fig["data"])):
        exp_traces.append(exp_fig["data"][trace])
    for trace in range(len(exp_stack_fig["data"])):
        exp_stack_traces.append(exp_stack_fig["data"][trace])
    
    main_fig = make_subplots(
        rows=1, #Half the amount of rows as factors, ceiling taken for whole number.
        cols=4,
       # horizontal_spacing=0.1,
       # vertical_spacing=0.02,
        subplot_titles=['AvE Trend','Gini Trend','Exposure Trend','Exposure Relative']
        )

    for traces in ave_traces:
        main_fig.append_trace(traces, row=1, col=1)
    for traces in gini_traces:
        main_fig.append_trace(traces, row=1, col=2)
    for traces in exp_traces:
        main_fig.append_trace(traces, row=1, col=3)
    
    main_fig.update_traces(showlegend=False)

    for traces in exp_stack_traces:
        main_fig.append_trace(traces, row=1, col=4)

    main_fig.update_layout(title_text=f"Summary of factor {factor_name}",
                  height=400, 
                  width=1440)
    main_fig.update_coloraxes(colorscale='Bluered',reversescale=True)

    main_fig.update_traces(histnorm='percent', 
                        histfunc='sum',
                        #bingroup='stack',
                        #alignmentgroup='1',
                        #offsetgroup='2',                     
        col=4)
    
    return main_fig

def make_overall_plotly_graph(dataframe):

    sorted_results=dataframe.sort_values(by=['Factor','Analysis_Week'])    
    overall_gini = sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'].values
    overall_mae = sorted_results['MAE'][sorted_results['Factor']=='Overall'].values
    sorted_results['AvE'] =  sorted_results['Actuals']/sorted_results['Predicted']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=sorted_results['Analysis_Week'][sorted_results['Factor']=='Overall'], 
                y=sorted_results['Gini_Norm'][sorted_results['Factor']=='Overall'], 
                name='Gini Norm Score',
                marker = {'color' : 'red'}
                ),
        secondary_y=True
    )

 #   fig.add_trace(
 #       go.Scatter(x=sorted_results['Analysis_Week'][sorted_results['Factor']=='Overall'], 
 #               y=sorted_results['MAE'][sorted_results['Factor']=='Overall'], 
 #               name='MAE',
 #               marker = {'color' : 'blue'}
 #               ),
 #       secondary_y=True
 #   )

    fig.add_trace(
        go.Scatter(x=sorted_results['Analysis_Week'][sorted_results['Factor']=='Overall'], 
                y=sorted_results['AvE'][sorted_results['Factor']=='Overall'],
                name='AvE',
                marker = {'color' : 'green'}
                ),
        secondary_y=True
    )

    fig.add_trace(
        go.Bar(x=sorted_results['Analysis_Week'][sorted_results['Factor']=='Overall'], 
                y=sorted_results['Count'][sorted_results['Factor']=='Overall'],  
                name="Quote Volume",
                marker = {'color' : 'yellow'}),
    )

    # Add figure title
    fig.update_layout(
        title_text="<b>Model Overall Performance</b>",
        
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Analysis Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="Count of quotes", 
                     secondary_y=False)
    fig.update_yaxes(#title_text="<b>Relativities</b>", 
                     secondary_y=True,
                     range=[0,2])
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## This is the first cell that you need to change. 
# MAGIC ###### Input the SQL table name here, actuals + prediction columns and then a list of columns to ignore. 
# MAGIC ###### If there are no columns to ignore then leave as []

# COMMAND ----------

output_df, time_output_df=model_drift_analysis(sql_dataset='pricing.home_market_model_drift_ld',
                         actuals_column='avgprice15',
                         predictions_column='predmarketprice', 
                         columns_to_ignore=['avgprice15','predmarketprice','uuid','hub_timestamp','quoteid','avgbrokerprice15','row_number','analysis_week'])

# COMMAND ----------

# Unsure if needed long term or just for this data.
# time_output_df['Analysis_Week']=time_output_df['Analysis_Week'].astype(float)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The below cell calculates the biggest movements per factor by AvE

# COMMAND ----------

# Create an empty dataframe that we will fill with the results
ordered_factors=pd.DataFrame(columns=['Factor','Score'])

# Create empty lists that we will populate with results, and then attach to the empty dataframe above.
factor_list=[]
score_list=[]

#For every factor in our dataset:
for factor in time_output_df['Factor'].unique():

    #Filter dataset so every level of the factor has more than 10000 exposure/rows 
    temp_df_=time_output_df[time_output_df['Factor']==factor][time_output_df['Count']>10000]

    #Attach the name of the factor to the factor_list.
    factor_list.append(factor)
    #Attach the difference between the maximum and minimum AvE of the levels.
    score_list.append((temp_df_["Actuals"]/temp_df_["Predicted"]).max()-(temp_df_["Actuals"]/temp_df_["Predicted"]).min())

#Attach lists back to dataframe
ordered_factors['Factor']=factor_list
ordered_factors['Score']=score_list

#Order the dataframe by our defined score above.
ordered_factors.sort_values('Score', ascending=False, inplace=True)

#Reset index of dataframe, this is more housekeeping style code rather than functional for us.
ordered_factors.reset_index(drop=True, inplace=True)


# COMMAND ----------

# MAGIC %md
# MAGIC #### This cell shows you the factors that have moved the most by AvE

# COMMAND ----------

ordered_factors

# COMMAND ----------

# MAGIC %md
# MAGIC # Below this cell are outputs used for creating the dashboard.
# MAGIC - Run each one and then on the top right handside of the cell click 'Add to Dashboard' 

# COMMAND ----------

# MAGIC %md
# MAGIC #####Dashboard Usage:
# MAGIC This dashboard is intended to give an overisight of the Home Market Model's performance over the last 6 weeks. \
# MAGIC AvE is calculated by taking the Actual expected value (avgprice15 from GoCo dataset), and diving it by the expected value (our market model prediction, predmarketprice from radar response.)
# MAGIC
# MAGIC - Note need to add other bits here

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC #### Factor Level Analysis:
# MAGIC Below are 4 graphs per factor used to determine if a factor is drifting. 
# MAGIC
# MAGIC The top 5 factors from analysis have been pulled out for special consideration. This was based on a 'Score' metric determined for each factor. 
# MAGIC
# MAGIC The score metric is calculated at factor level by taking the difference between the minimum and maximum AvE value across the weeks. This helps highlight any big swings in movement against the market.
# MAGIC
# MAGIC ---

# COMMAND ----------

make_factor_summary_plotly_graph(time_output_df, 
                                 ordered_factors['Factor'][0])

# COMMAND ----------

make_factor_summary_plotly_graph(time_output_df, 
                                 ordered_factors['Factor'][1])

# COMMAND ----------

make_factor_summary_plotly_graph(time_output_df, 
                                 ordered_factors['Factor'][2])

# COMMAND ----------

make_factor_summary_plotly_graph(time_output_df, 
                                 ordered_factors['Factor'][3])

# COMMAND ----------

make_factor_summary_plotly_graph(time_output_df, 
                                 ordered_factors['Factor'][4])

# COMMAND ----------

make_overall_plotly_graph(time_output_df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


# Code to output any amount of graphs into 2 columns and 1 display.


fig = make_subplots(
    rows=int(np.ceil(len(factors_to_monitor)/2)), #Half the amount of rows as factors, ceiling taken for whole number.
    cols=2,
    horizontal_spacing=0.1,
    vertical_spacing=0.02,
    subplot_titles=(factors_to_monitor),
    specs=list_of_lists
    )


for i, factor in enumerate(factors_to_monitor): #enumerate here to get access to i

    # Create figure with secondary y-axis


    # Add traces
    fig.add_trace(
        go.Scatter(x=filtered_results['Level'][filtered_results['Factor']==factor], 
                y=filtered_results['Gini'][filtered_results['Factor']==factor]/overall_gini, 
                name='Gini Score',
                legendgroup=f'{factor}',
                marker = {'color' : 'red'}

                ),
    
        row=int(np.ceil(i)/2)+1, 
        col=i%2 +1,
        secondary_y=True

    )

    fig.add_trace(
        go.Scatter(x=filtered_results['Level'][filtered_results['Factor']==factor], 
                y=filtered_results['MAE'][filtered_results['Factor']==factor]/overall_mae, 
                name='MAE',
                legendgroup=f'{factor}',
                marker = {'color' : 'blue'}
                ),
        row=int(np.ceil(i)/2)+1, 
        col=i%2 +1,
        secondary_y=True

    )

    fig.add_trace(
        go.Bar(x=filtered_results['Level'][filtered_results['Factor']==factor], 
                y=filtered_results['Count'][filtered_results['Factor']==factor],  
                name="Quote Volume",
                legendgroup=f'{factor}',
                marker = {'color' : 'yellow'}
                ),
        row=int(np.ceil(i)/2)+1, 
        col=i%2 +1,
        secondary_y=False
    )


    # Set y-axes titles
    #fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
    #fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig.update_layout(title_text="Relative Gini and MAE scores per Factor.",
                  height=len(factors_to_monitor)*300, 
                  width=1440,
                  legend_tracegroupgap = 430)
fig.add_hline(y=1, line_dash="dot", row="all", col="all",
              annotation_text="1 Baseline", 
              annotation_position="bottom right",
              yref = 'y')

fig.show()
