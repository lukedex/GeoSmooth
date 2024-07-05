import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

# TO DO: 
# change colours of boxplots to match x-graphs AND/OR alter plots for violin plots
# add plotted gini curves
# change to df output and indicate winning model using pandas

def gini_table(df, pred, act):
    
    ''' returns a dataframe from which a gini coefficient can be calculated
        also can create cumulative gains curves
        pred = predicted values (output from emblem) argument required is the name of the column in df
        act = actual values (number of claims) argument required is the name of the column in df
    
        3 useful outputs
        Perc of Obs and Perc of Claims can be used to create Cumulative Gains Curves
        Gini_Area can be used to calculate the gini coefficient. Each Gini_Area is the approximate area under Cumulative
        gains curve. Feel free to change to trapezium rule in future. '''
    
    df = df[[pred, act]].sort_values(by=pred, ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df[act].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    return df

def calc_gini(df, pred, act):
    
    ''' uses output from gini_table to calculate a gini coefficient. Formula comes from R:\Pricing\Personal Lines Pricing - Motor\Technical\21. Provident\
        4. SAS Processes\Technical MI Tools\Gini_Coefficients_and_U_Statistics\1.Motivation - GiniCoefficientpaper.pdf
        model = column name of modelled values you wish to calculate gini coefficient of.
        obs = column name of actual values (number of claims) '''
    
    d1 = gini_table(df, pred, act)
    Gini_coef = round((d1.sum()['gini_area'] - 0.5) *2,6)
    return(Gini_coef)

def gini_coef(df, pred, act):
    
    ''' function to calculate gini using input df, prediction and actuals
        values must be rebased and exposure weighted before use 
        derived from gini_table and calc_gini '''

    df = df[[pred, act]].sort_values(by=pred, ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df[act].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    Gini_coef = round((df.sum()['gini_area'] - 0.5) *2,6)

    return Gini_coef

def exposure_weighting(df, col, exposure):

    ''' function to create exposure weighted output column for any given input df, column and exposure '''

    df[col+'_ew']=df[col]*df[exposure]
    return df[col+'_ew'] 

def rebase(df, col, actuals):

    ''' function to rebase for any given input df, column and actual prediction '''

    anchor_total=df[actuals].sum()
    col_total=df[col].sum()
    rebase_mult=anchor_total/col_total
    df[col+'_rebase']=df[col]*rebase_mult
    return df[col+'_rebase']

def downsample_func(df, fraction=1):

    ''' function to downsample dataset with replacement for any given input df and specified fraction
        default fraction set to 1 '''

    # randomly select data with replacement
    df=df.sample(frac = fraction, replace = True)
    return df

def Average(lst):

    ''' function to create the average value from a list '''

    return sum(lst) / len(lst)

def down_gini(df, pred1, pred2, act, n):

    ''' function which creates n downsampled datasets and calculates original gini, average gini and 5%/95% confidence intervals 
        using a given input df, prediction and actual. '''

    pred1_original_gini=round(gini_coef(df, pred1, act),3)
    pred2_original_gini=round(gini_coef(df, pred2, act),3)
    pred1_ginis=[]
    pred2_ginis=[]

    for i in range(1, n):
            
        # create a downsampled dataframe
        down_df=downsample_func(df)
        
        # calculate gini on sample
        pred1_gini=gini_coef(down_df, pred1, act)
        pred2_gini=gini_coef(down_df, pred2, act)        
        pred1_ginis.append(pred1_gini)
        pred2_ginis.append(pred2_gini)
                    
    # convert list into an array
    pred1_a=np.array(pred1_ginis)
    pred2_a=np.array(pred2_ginis)
    
    # calculate gini and confidence intervals
    pred1_p5 = round(np.percentile(pred1_a, 5),3)
    pred1_p50 = round(np.percentile(pred1_a, 50),3)
    pred1_p95 = round(np.percentile(pred1_a, 95),3)
    pred2_p5 = round(np.percentile(pred2_a, 5),3)
    pred2_p50 = round(np.percentile(pred2_a, 50),3)
    pred2_p95 = round(np.percentile(pred2_a, 95),3)

    
    return pred1_original_gini, pred1_p5, pred1_p50, pred1_p95, pred1_ginis, \
pred2_original_gini, pred2_p5, pred2_p50, pred2_p95, pred2_ginis

def box_plot_gini(data, n):

    ''' function to create a boxplots of model gini values for an input list of ginis and the number of interations n '''

    # defining axes
    fig = plt.figure(figsize =(10, 10))
    #fig = plt.rcParams["figure.figsize"] = (10,10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title(f'Gini Coefficients built on {n} sampled datasets')
    ax.set_ylabel('Gini Coefficient')
    ax.set_xlabel('Models')
    ax.yaxis.grid(True, linestyle='dashed', dashes=(5, 10), linewidth=0.5) # horizontal gridlines
    ax.xaxis.grid(False) # vertical gridlines
    labels = ['model 1', 'model 2']

    # create plot
    bp = ax.boxplot(data, labels=labels)
    
    # show plot
    plt.show()

def create_xgraph_inputs(unique_id, new_model_predictions, old_model_predictions, actuals, input_df):

    ''' function to create inputs for x-graph ''' 

    cols=[unique_id, new_model_predictions, old_model_predictions, actuals]
    input_df=input_df[cols].copy()

    input_df['Diff']=(input_df[new_model_predictions]/input_df[old_model_predictions])-1
    
    # remove inf values
    input_df=input_df[input_df['Diff']!=np.inf]

    # x value bandings (may need to be tweaked dependent on the prediction values)
    input_df['Band']=round(input_df['Diff'],3)
    #input_df['Band']=np.floor(input_df['Diff']*5)/5
    #df['Band']=round(df['Band'],1)
    #df['Diff'] = pd.cut(df['Diff'],bins=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]).apply(lambda x : x.right)
    #input_df['Band']=pd.qcut(input_df['Band'].rank(method='first'), 10).apply(lambda x : x.right)
    #input_df['Band']=input_df.groupby('Band')['Diff'].transform('mean')
    #input_df['Band']=round(input_df['Band'],4)
    #df['Band']=pd.qcut(df['Diff'], q=1, precision=0).apply(lambda x : x.right)

    # create Pivot Table    
    pivot=input_df.groupby('Band').size().reset_index(name='Count')
    grouping=input_df.groupby('Band').sum().reset_index()
    cols=['Band', 'Count', new_model_predictions, old_model_predictions, actuals]
    pivot=pivot.merge(grouping, on='Band', how='left')[cols]
    pivot['Rank']=pivot['Band'].rank(method='dense')

    # summarisation
    constant=pivot[actuals].sum()/pivot[old_model_predictions].sum()
    pivot['Current Model']=(pivot[old_model_predictions]/pivot['Count'])*constant

    constant=pivot[actuals].sum()/pivot[new_model_predictions].sum()
    pivot['New Model']=(pivot[new_model_predictions]/pivot['Count'])*constant

    pivot['Observed Conversion']=pivot[actuals]/pivot['Count']
    pivot['Squared Error Old']=pow(pivot['Current Model']-pivot['Observed Conversion'],2)*pivot['Count']
    pivot['Squared Error New']=pow(pivot['New Model']-pivot['Observed Conversion'],2)*pivot['Count']

    # output
    print('Squared Error Pred 1:', round(pivot['Squared Error Old'].sum(),3))
    print('Squared Error Pred 2:', round(pivot['Squared Error New'].sum(),3))

    return pivot

def create_xgraph(exposure, prop, curr, obs, diff_band, rank, new_model_predictions, old_model_predictions):

    ''' create Emblem style x-graph '''

    # Bar plot
    plt.rcParams["figure.figsize"] = (10,10)
    N=len(exposure)
    width = 0.5
    plt.bar(rank, exposure, width, color='gold', label='Diff', edgecolor='k')
    plt.xticks(rank, diff_band, rotation=90)
    plt.ylim(0, max(exposure)*3)
    plt.ylabel('Exposure')
    plt.xlabel(f'Banded Difference')
    plt.title(f'X-Graph')

    # Line plot
    axes2 = plt.twinx()
    axes2.plot(rank, obs, color='fuchsia', marker="s", markeredgecolor='black', label='Actual')
    axes2.plot(rank, prop, color='blue', marker="D", markeredgecolor='black', label=f'{new_model_predictions} Model')
    axes2.plot(rank, curr, color='g', marker="^", markeredgecolor='black', label=f'{old_model_predictions} Model')
    axes2.set_ylabel('Average Response per Exposure Bucket')

    # legend and settings
    plt.legend(loc="upper left")

    plt.show()

def create_rebased_xgraph(exposure, prop, curr, obs, diff_band, rank, new_model_predictions, old_model_predictions):

    ''' create rebased Emblem style x-graph '''

    # rebase based on current predictions
    prop=prop/curr
    obs=obs/curr
    curr=curr/curr

    # Bar plot
    plt.rcParams["figure.figsize"] = (10,10)
    N=len(exposure)
    width = 0.5
    plt.bar(rank, exposure, width, color='gold', label='Diff', edgecolor='k')
    plt.xticks(rank, diff_band, rotation=90)
    plt.ylim(0, max(exposure)*3)
    plt.ylabel('Exposure')
    plt.xlabel(f'Banded Difference')
    plt.title(f'X-Graph')

    # Line plot
    axes2 = plt.twinx()
    axes2.plot(rank, obs, color='fuchsia', marker="s", markeredgecolor='black', label='Actual')
    axes2.plot(rank, prop, color='blue', marker="D", markeredgecolor='black', label=f'{new_model_predictions} Model')
    axes2.plot(rank, curr, color='g', marker="^", markeredgecolor='black', label=f'{old_model_predictions} Model')
    axes2.set_ylabel('Average Response per Exposure Bucket')

    # legend and settings
    plt.legend(loc="upper left")

    plt.show()

def performance_metrics(df, pred1, pred2, actual):
    
    ''' function to return RMSE, MAE and MSLE for input predictions'''
    
    # RMSE
    pred1_rms = round(mean_squared_error(df[actual], df[pred1], squared=False),2)
    print('Pred 1 RMSE for Original dataset: ', pred1_rms)
    pred2_rms = round(mean_squared_error(df[actual], df[pred2], squared=False),2)
    print('Model 2 RMSE for Original dataset: ', pred1_rms, '\n')    
    
    # MAE
    pred1_mae = round(mean_absolute_error(df[actual], df[pred1]),2)
    print('Model 1 MAE for Original dataset: ', pred1_mae)
    pred2_mae = round(mean_absolute_error(df[actual], df[pred2]),2)
    print('Model 2 MAE for Original dataset: ', pred2_mae, '\n')   
    
    # MSLE
    pred1_msle = round(mean_squared_log_error(df[actual], df[pred1]),3)
    print('Model 1 MSLE for Original dataset: ', pred1_msle)
    pred2_msle = round(mean_squared_log_error(df[actual], df[pred2]),3)
    print('Model 2 MSLE for Original dataset: ', pred2_msle, '\n')
    
def lift_chart(df, pred1, pred2, actual):
    
    ''' module to calculate and plot single lift chart '''

    # sort values by actual (methodologies differ)
    df=df.sort_values(by=actual)

    # create 20 different exposure buckets
    df['Band'] = pd.qcut(df[actual], q=20, precision=0).apply(lambda x : x.left)

    # create inputs to plot
    exposure=df.groupby('Band').size().reset_index(name='Count')['Count']
    curr=df.groupby('Band')[pred1].mean().reset_index(name='Pred 1 Average')['Pred 1 Average']
    prop=df.groupby('Band')[pred2].mean().reset_index(name='Pred 2 Average')['Pred 2 Average']
    obs=df.groupby('Band')[actual].mean().reset_index(name='Actual Average')['Actual Average']
    rank=df.groupby('Band').size().reset_index(name='Count')['Band'].rank(method='dense')

    # Bar plot
    plt.rcParams["figure.figsize"] = (10,10)
    N=len(exposure)
    width = 0.5
    plt.bar(rank, exposure, width, color='gold', label='Diff', edgecolor='k')
    # plt.xticks(rank, diff_band, rotation=90)
    plt.ylim(0, max(exposure)*3)
    plt.ylabel('Exposure')
    plt.xlabel(f'Exposure Bucket')
    plt.title(f'Lift Chart')

    # Line plot
    axes2 = plt.twinx()
    axes2.plot(rank, obs, color='fuchsia', marker="s", markeredgecolor='black', label='Actual')
    axes2.plot(rank, prop, color='blue', marker="D", markeredgecolor='black', label=f'{pred2} Model')
    axes2.plot(rank, curr, color='g', marker="^", markeredgecolor='black', label=f'{pred1} Model')
    axes2.set_ylabel('Average Response per Exposure Bucket')

    # legend and settings
    plt.legend(loc="upper left")

    plt.show()    
    
def model_validation(df, pred1, pred2, act, unique_id, n=10):

    ''' function to run gini and x-graphs'''

    print('MODEL VALIDATION RESULTS', '\n')

    # SUMMARY STATISTICS
    print('Summary Statistics of Predictions:', '\n')
    display(df[[pred1, pred2, act]].describe().reset_index())
    print('\n')

    # GINI
    val1=down_gini(df, pred1, pred2, act, n)

    print(f'Pred 1 Gini for Original dataset: ', val1[0])
    print(f'Pred 2 Gini for Original dataset: ', val1[5], '\n')

    print(f'Pred 1 5% CI Gini for {n} datasets: ', val1[1])
    print(f'Pred 2 5% CI Gini for {n} datasets: ', val1[6], '\n')

    print(f'Pred 1 Average Gini for {n} datasets: ', val1[2])
    print(f'Pred 2 Average Gini for {n} datasets: ', val1[7], '\n')

    print(f'Pred 1 95% CI Gini for {n} datasets: ', val1[3])
    print(f'Pred 2 95% CI Gini for {n} datasets: ', val1[8], '\n')

    print('Pred 2 Model Improvement (on Original): ', round(((val1[5]/val1[0])-1)*100,2))
    print(f'Pred 2 Model Improvement (over {n} iterations): ', round(((val1[7]/val1[2])-1)*100,2))

    # plot Gini
    box_plot_gini([val1[4], val1[9]], n)

    # X-GRAPH
    pivot=create_xgraph_inputs(unique_id, pred2, pred1, act, df)
    exposure=pivot['Count']
    prop=pivot['New Model']
    curr=pivot['Current Model']
    obs=pivot['Observed Conversion']
    diff_band=pivot['Band']
    rank=pivot['Rank']

    # plot X-Graph
    create_xgraph(exposure, prop, curr, obs, diff_band, rank, pred2, pred1)
    create_rebased_xgraph(exposure, prop, curr, obs, diff_band, rank, pred2, pred1)
    
    # Lift Chart
    #lift_chart(df, pred1, pred2, act)