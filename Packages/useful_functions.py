import pandas as pd
import numpy as np
import psutil
import gc


def mem_use():
    # Get the memory usage of the notebook
    memory_usage = psutil.Process().memory_info().rss

    # Convert the memory usage to megabytes
    memory_usage_mb = memory_usage / 1024 / 1024

    # Print the memory usage
    print(f"Memory usage: {memory_usage_mb:.2f} MB")


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


# Gini2 has been altered to work with xgboost.
def gini2(preds, dtrain):
    actual = dtrain.get_label()
    assert(len(actual) == len(preds))
    expo = np.ones(len(actual))
    all1 = np.asarray(np.c_[ actual, preds, np.linspace(0, 1, len(y_val)),expo], dtype=float)
    all = all1[ np.lexsort((all1[:,2], -1*all1[:,1])) ]
    actTotal = all[:,0].sum()
    expoTotal = all1[:,3].sum()
    actCumSum = all[:,0].cumsum()
    expoCumSum = all1[:,3].cumsum()
    g = (actCumSum*all1[:,3]).sum()/(actTotal*expoTotal)
    return 'Gini', 2*(g - 0.5)

def test_func():
    print("Hello World!")
    
# This function downsamples training data, and upweights accordingly. Returns two dataframes.
def downsampling(X_train, y_train, frac, upweight = None, seed = 42):
    
    #create local copies so doesn't edit original data
    local_df = X_train.copy()
    local_df['response'] = y_train.copy()
    
    #split into positive and negative datasets
    pos_data = local_df[local_df['response'] > 0].copy()
    neg_data = local_df[local_df['response'] <= 0].copy()
    
    #random sample of length n, upweight exposure by float
    down_data = neg_data.sample(frac = frac, random_state = seed)
    #down_data = neg_data[neg_data['Random_10'] < frac].copy()
    
    if upweight == None:
        upweight = 1/frac
    
    down_data['Exposure'] *= upweight
    
    #join downsampled negative data with pos data
    down_data = pd.concat([down_data, pos_data]).sample(frac = 1)
    
    #remove response for X_train, y_train becomes downsampled response
    X_train_return = down_data.drop('response', axis=1)
    y_train_return = down_data['response']
    
    return X_train_return, y_train_return

def downsample_func(df, fraction=1):

    ''' function to downsample dataset with replacement for any given input df and specified fraction
        default fraction set to 1 '''

    # randomly select data with replacement
    df=df.sample(frac = fraction, replace = True)
    return df

def down_gini(pred, act, n):

    ''' function which creates n downsampled datasets and calculates original gini, average gini and 5%/95% confidence intervals 
        using a given input df, prediction and actual. '''

    gini_list = []

    df_local = pd.DataFrame(list(zip(pred, act)), columns = ['Predicted', 'Actual'])
    
    for i in range(n):
            
        # create a downsampled dataframe
        down_df=downsample_func(df_local)
        # calculate gini on sample
        pred1_gini= gini(down_df['Predicted'], down_df['Actual'])  
        gini_list.append(pred1_gini)

    return gini_list