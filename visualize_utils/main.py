import json
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm

from mesurement_metric import mesurement_metric
from data_loader import RawDataLoader

if __name__ == "__main__":
    # X = np.array([[1, 2], [1, 4], [1, 0],
    #             [10, 2], [10, 4], [10, 0]])
    # y =  np.array([1,1,1,0,1,0])
    data_folder = 'data/test_data_04_20'
    raw_data_loader = RawDataLoader(data_folder)
    print(raw_data_loader.df)
    
    
    all_setting_maps = [
        {'direction': ['straight', 'reverse', 'forward'], 
         'p': ['P = 50', 'P = 40', 'P = 30', 'P = 20'], 
         'area_type': ['even', 'odd'], 
         'features_size': [1000, 100, 200],
         'start_pos': [i for i in range(0, 1000, 100)]},
        {'direction': ['original'], 
         'p': ['P = 0'], 
         'area_type': ['even', 'odd'], 
         'features_size': [1000, 100, 200], 
         'start_pos': [i for i in range(0, 1000, 100)]} 
    ]
    all_settings = []
    for setting_map in all_setting_maps:
        tmp_setting = []
        for setting in itertools.product(*setting_map.values()):
            all_settings.append(dict(zip(setting_map.keys(), setting)))
            
    all_results = []
    all_info = []
    for idx, setting in enumerate(tqdm(all_settings)):
        infor = {}
        cur_setting_result = {}
        cur_setting_result['setting'] = setting
        
        # filter data 
        dat_filtered = raw_data_loader.filter(**setting)
        
        # eval score
        result_scores = mesurement_metric(dat_filtered['value'].values.tolist(), dat_filtered['label'].values, n_clusters=3)
        cur_setting_result['result'] = result_scores
        
        # save result
        all_results.append(cur_setting_result)
        all_info.append({ **setting, **result_scores})
        if idx % 10 == 0:
            json.dump(all_results, open(f'{data_folder}/all_results.json', 'wt'), indent=2)
            
    json.dump(all_results, open(f'{data_folder}/all_results.json', 'wt'), indent=2)
    df_info = pd.DataFrame(all_info)
    df_info.to_csv(open(f'{data_folder}/all_results.csv', 'wt'))
     
    