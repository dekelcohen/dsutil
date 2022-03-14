# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:34:40 2022

@author: DEKELCO
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree 

class SimilarSamples:
    def __init__(self, df_data, columns=None):
        self.cols = df_data.columns if not columns else columns
        self.df_data = df_data.reset_index(drop=True)
        self.tree = BallTree(self.df_data[self.cols].values)
        
    def __get_similar_rows_by_k(self,df_query,k=1):
        dist, ind = self.tree.query(df_query.values, k)
        return dist, ind
    
    def __get_similar_rows_by_radius(self,df_query,radius=1,count_only=False):    
        if count_only:
            return self.tree.query_radius(df_query.values, radius, count_only=True)
        
        ind, dist = self.tree.query_radius(df_query.values, radius, return_distance=True, count_only=False)
        return dist, ind

    def merge_similar_rows(self,df_query, k=1, radius=None):        
        df_query = df_query.copy()        
        if radius:
            dist, ind = self.__get_similar_rows_by_radius(df_query[self.cols],radius)
        else:
            dist, ind = self.__get_similar_rows_by_k(df_query[self.cols],k)
        
        df_query['matched_idx'] = ind.tolist()
        df_query = df_query.explode('matched_idx')
        df_query['match_list'] = np.concatenate(dist.tolist()) 
        df_merged = pd.merge(df_query,self.df_data,left_on='matched_idx',right_index=True)
        return df_merged
    
    def count_similar_rows(self,df_query, radius=0):        
        count = self.__get_similar_rows_by_radius(df_query[self.cols], radius, count_only=True)
        return count
    

def test():
    df_data = pd.DataFrame(data=[{'age' : 24,'gender' : 1}, {'age' : 32,'gender' : 1}, {'age' : 72,'gender' : 1}, {'age' : 34,'gender' : 0}])
    df_query = pd.DataFrame(data=[{'age' : 25,'gender' : 1}, {'age' : 70,'gender' : 1}, {'age' : 47,'gender' : 0}])
    ss = SimilarSamples(df_data)
    df_merged = ss.merge_similar_rows(df_query)
    df_merged.head(10)
    df_merged = ss.merge_similar_rows(df_query, k=2)
    df_merged.head(10)
    
if __name__ == "__main__":
    test()