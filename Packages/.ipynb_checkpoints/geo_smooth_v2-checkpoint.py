import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
import os


class geo_smoothing:
    def __init__(self, 
                 gdf, values_col, 
                 method, KNN_Num, 
                 own_weight_val, 
                 max_iterations,
                 exposure_weight, exposure_col, 
                 outputs, output_folder_name):
        
        self.gdf=gdf
        self.values_col=values_col
        self.method=method
        self.KNN_Num=KNN_Num
        self.own_weight_val=own_weight_val
        self.max_iterations=max_iterations
        self.exposure_weight=exposure_weight
        self.exposure_col=exposure_col
        self.outputs=outputs
        self.output_folder_name=output_folder_name
        self.iter_round=0

    def neighbour_weight_setting(self):

        PATH = f'GS_Outputs/{self.output_folder_name}/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        if self.method == 'KNN':
            W = libpysal.weights.KNN.from_dataframe(self.gdf, 
                                                    k=self.KNN_Num, 
                                                    silence_warnings=True)
        elif self.method == 'Rook':    
            W = libpysal.weights.Rook.from_dataframe(self.gdf, 
                                                     silence_warnings=True)
        elif self.method == 'Queen':
            W = libpysal.weights.Queen.from_dataframe(self.gdf, 
                                                      silence_warnings=True)
            print(f'{len(W.islands)} islands found. Attaching nearest {self.KNN_Num} neighbours')
            W_knn = libpysal.weights.KNN.from_dataframe(self.gdf, 
                                                    k=self.KNN_Num, 
                                                    silence_warnings=True)
            W = libpysal.weights.w_union(W, W_knn)
            print(f'{len(W.islands)} islands remain.')
        else:
            print("Method not found, defaulting to Queen")
            W = libpysal.weights.Queen.from_dataframe(self.gdf, 
                                                      silence_warnings=True)

        if self.exposure_weight == True:
            ## This will take the Exposure from an 'Exposure' column and weight predictions by it
            Exposure_map = pd.Series(self.gdf[self.exposure_col].values,
                                     index=self.gdf.index).to_dict()

            for i in range(0,len(W.neighbors)):
                #Add itself to it's own neighbours to include own cell weightage and target
                W.neighbors[i].append(i)
                #Adds a value of to the weights to represnt itself (chosen to be dynamic)
                #Takes the maximum weight seen and uses that multipled by the own_weight_val
                W.weights[i].append(max(W.weights[i], 
                                        default=1)*self.own_weight_val)
                exp_vals=list(map(Exposure_map.get,
                                  W.neighbors[i]))
                W.weights[i] = [a*b for a,b in zip(exp_vals,W.weights[i])]
        ## Transform weights so that sum(neighbours) = 1        
        W.transform = "r"
        self.weights=W
        
        
    
    def spatial_lag(self):
        print('Running base smoothing')
        self.gdf[f"smoothed_value_i0"] = libpysal.weights.lag_spatial(self.weights, 
                                                                   self.gdf[self.values_col])
        self.gdf[f"smoothed_value_i0"] = np.where(self.gdf["smoothed_value_i0"]==0, 
                                               self.gdf[self.values_col], 
                                               self.gdf["smoothed_value_i0"])
    def spatial_lag_iterator(self):
        print(f'Running {self.iter_round +1} round of smoothing')
        self.gdf[f"smoothed_value_i{self.iter_round+1}"] = libpysal.weights.lag_spatial(self.weights, 
                                                                   self.gdf[f'smoothed_value_i{self.iter_round}'])
        self.iter_round+=1      
    
    def smoothing_rounds(self):
        geo_smoothing.spatial_lag(self)
        geo_smoothing.create_outputs(self)
        #self.iter_round+=1
        while self.iter_round < self.max_iterations:
            geo_smoothing.spatial_lag_iterator(self)
            geo_smoothing.create_outputs(self)
        
    def create_outputs(self):
        if self.outputs==True:
            self.gdf.explore(column = f"smoothed_value_i{self.iter_round}", 
                             tooltip = f"smoothed_value_i{self.iter_round}",
                             style_kwds=dict(weight=0.5, opacity=0.8, fillOpacity=0.8)
                            ).save(f'GS_Outputs/{self.output_folder_name}/{self.method}_smooth_i{self.iter_round}.html')
            
            if self.iter_round==0:                                   
                self.gdf.explore(column = self.values_col, 
                                 tooltip = self.values_col,
                             style_kwds=dict(weight=0.5, opacity=0.8, fillOpacity=0.8)
                                ).save(f'GS_Outputs/{self.output_folder_name}/{self.method}_raw.html')
                                 
    def run(self):
        geo_smoothing.neighbour_weight_setting(self)
        geo_smoothing.smoothing_rounds(self)
        return self.gdf            