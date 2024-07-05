import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
import os

def distance_smoothing(gdf, 
                       values_col, 
                       distance, 
                       alpha_val = -1,
                       exposure_weight=False,
                       exposure_col=None,
                       outputs=False,
                       silent=True,
                      own_weight_val=1):
    """
    Function will perform a distance smoothing algorithm based on a geometric dataframe (GDF)
    
    Inputs:
        gdf :  Geometric DataFrame with a column called 'geometry'
        values_col : Values column to be smoother
        distance : Distance threshold for neighbours 
        alpha : decay factor when negative. Normal value is -1
        exposure_weight : Boolean, if to use exposure weighted postcode weightings
        exposure_col : Column used for exposure
        outputs : Boolean, if to create GS_Outputs folder and populate with results
        
    Outputs:
        Decile ensembles graphs outputted in HTML format.
        Decile ensembles dataframe attached to gdf.
    """
    import os
    
    #Creates a folder called 'GS_Outputs'
    PATH = 'GS_Outputs/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    W_distance = libpysal.weights.DistanceBand.from_dataframe(gdf,
                                                         threshold=distance,
                                                         distance_metric="euclidean", #Better than Manhattan
                                                         binary=False,
                                                         alpha=alpha_val,
                                                         silence_warnings=silent)

    if exposure_weight == True:
        ## This will take the Exposure from an 'Exposure' column and weight predictions by it
        Exposure_map = pd.Series(gdf[exposure_col].values,index=gdf.index).to_dict()
        
        for i in range(0,len(W_distance.neighbors)):
            
            #Add itself to it's own neighbours to include own cell weightage and target
            W_distance.neighbors[i].append(i)
            #Adds a value of to the weights to represnt itself (chosen to be dynamic)
            #Takes the maximum weight seen and uses that multipled by the own_weight_val
            W_distance.weights[i].append(max(W_distance.weights[i], default=1)*own_weight_val)
            
            #Times exposure to the weights
            exp_vals=list(map(Exposure_map.get, W_distance.neighbors[i]))
            W_distance.weights[i] = [a*b for a,b in zip(exp_vals,W_distance.weights[i])]
        
        
    ## Transform weights so that sum(neighbours) = 1
    W_distance.transform = "r"                    
    
    ## Perform spatial lag
    gdf["dist_smooth_100"] = libpysal.weights.lag_spatial(W_distance, gdf[values_col])
    
        
    #Makes the average returned smoothed value equal to the average value input
    gdf["dist_smooth_100"] = gdf["dist_smooth_100"]*gdf[values_col].mean()/gdf["dist_smooth_100"].mean()
    

    if outputs==True:
        gdf.explore(column = "dist_smooth_100", tooltip = 'dist_smooth_100').save("GS_Outputs/dist_smooth_100.html")
        gdf.explore(column = values_col, tooltip = 'POSTSECT').save("GS_Outputs/dist_no_smoothing.html")
        
    return gdf

def neighbour_smoothing(gdf, 
                        values_col, 
                         method,
                         exposure_weight=False,
                         exposure_col=None,
                         outputs=False,
                         KNN_Num=None,
                         own_weight_val=1):
    """
    Function will perform a distance smoothing algorithm based on a geometric dataframe (GDF)
    
    Inputs:
        gdf :  Geometric DataFrame with a column called 'geometry'
        values_col : Values column to be smoother
        method : Rook / Queen or KNN
        outputs : Boolean, if to create GS_Outputs folder and populate with results
        exposure_col :
        KNN_Num : Amount of K-nearest neighbours to use for KNN method
    Outputs:
        Decile ensembles graphs outputted in HTML format.
        Decile ensembles dataframe attached to gdf.
    """
    
    
    PATH = 'GS_Outputs/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    if method == 'KNN':
        W = libpysal.weights.KNN.from_dataframe(gdf,
                                               k=KNN_Num)
        
    elif method == 'Rook':    
        W = libpysal.weights.Rook.from_dataframe(gdf)
    elif method == 'Queen':
        W = libpysal.weights.Queen.from_dataframe(gdf,
                                                 silence_warnings=True)
    else:
        print("Method not found, defaulting to Queen")
        W = libpysal.weights.Queen.from_dataframe(gdf)
    

    if exposure_weight == True:
        ## This will take the Exposure from an 'Exposure' column and weight predictions by it
        Exposure_map = pd.Series(gdf[exposure_col].values,index=gdf.index).to_dict()

        
        for i in range(0,len(W.neighbors)):
            
            #Add itself to it's own neighbours to include own cell weightage and target
            W.neighbors[i].append(i)
            #Adds a value of to the weights to represnt itself (chosen to be dynamic)
            #Takes the maximum weight seen and uses that multipled by the own_weight_val
            W.weights[i].append(max(W.weights[i], default=1)*own_weight_val)
            
            
            exp_vals=list(map(Exposure_map.get,W.neighbors[i]))
            #sum_vals = sum(exp_vals)
            #stand_vals = [x / sum_vals for x in exp_vals]
            W.weights[i] = [a*b for a,b in zip(exp_vals,W.weights[i])]
        
        
        
    ## Transform weights so that sum(neighbours) = 1        
    W.transform = "r"

        
        
    gdf[f"{method}_smooth_100"] = libpysal.weights.lag_spatial(W, gdf[values_col])
    
    gdf[f"{method}_smooth_100"] = np.where(gdf[f"{method}_smooth_100"]==0, 
                                           gdf[values_col], 
                                           gdf[f"{method}_smooth_100"])
    
    
    if outputs==True:
        gdf.explore(column = f"{method}_smooth_100", tooltip = f"{method}_smooth_100").save(f"GS_Outputs/{method}_smooth_100.html")
        gdf.explore(column = values_col, tooltip = values_col).save(f"GS_Outputs/{method}_no_smoothing.html")
    
    return gdf    