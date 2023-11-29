import os
from glob import glob
import numpy as np 
import pandas as pd 
import pickle 

class RawDataLoader():
    def __init__(self, path_data_folder) -> None:
        if os.path.exists(f"{path_data_folder}/all_data.pkl"):
            self.df = pickle.load(open(f"{path_data_folder}/all_data.pkl", 'rb'))
        else:
            self.load_data_all_settings(path_data_folder)
            pickle.dump(self.df, open(f"{path_data_folder}/all_data.pkl", 'wb'))
            
        
    def load_data_all_settings(self, data_folder, n_skip_row=13):
        data_frame_raw = {
            'file_name': [],
            'direction': [],
            'p': [],
            'area_type': [],
            'value': [],
            'label': [],
        }
        for file_name in glob(f'{data_folder}{os.path.sep}**{os.path.sep}*.CSV', recursive=True):
            attritbute_values = [e for e in file_name.replace(data_folder, "").split(os.path.sep) if len(e) > 0]
            # only take 2 attr values in path 
            attritbute_values = attritbute_values[:2] # in (straight, P = 30, P80_c0000_I0019.CSV) => out (straight, P = 30)
            
            # check odd or even in file name
            idx = int(file_name[:-4].split("_")[-1][-1:])
            if idx % 2 == 1:
                area_type = 'odd'
            else:
                area_type = 'even'
                
            # load data content of one sammple 
            data_sample = self.load_data_sample(file_name, n_skip_row=n_skip_row)
            
            # save as data frame 
            data_frame_raw['file_name'].append(file_name)
            data_frame_raw['direction'].append(attritbute_values[0])
            data_frame_raw['p'].append(attritbute_values[1])
            data_frame_raw['area_type'].append(area_type)
            data_frame_raw['value'].append(data_sample['values'])
            data_frame_raw['label'].append(data_sample['label'])

        self.df = pd.DataFrame(data_frame_raw)
        
    def load_data_sample(self, file_name, n_skip_row):
        data_sample = {'label': None, 'values': None, 'file_name': file_name} 
            
        # load data to get label
        label = file_name.split(os.path.sep)[-1].split("_")[0] # input (data/test_data_04_20/forward/P = 50/P80_c0000_I0019.CSV) => output (P80)
        data_sample['label'] = label

        # load real values
        df = pd.read_csv(open(file_name), skiprows=n_skip_row, header=0)

        def filter_fn(row):
            return row

        df = df.apply(filter_fn, axis=1)
        values = df[df.columns[1]].tolist()
        data_sample['values'] = np.array(values)
        
        return data_sample
    
    @staticmethod
    def _df_limit_features(row, features_size, start_pos):
        row['value'] = row['value'][start_pos:start_pos+features_size]
        return row 
    
    def filter(self, direction=None, p=None, area_type=None, features_size=None, start_pos=0):
        df_selected = self.df 
        if direction is not None:
            df_selected = df_selected[df_selected['direction'] == direction]
        if p is not None:
            df_selected = df_selected[df_selected['p'] == p]
        if area_type is not None:
            df_selected = df_selected[df_selected['area_type'] == area_type]
        if features_size is not None:
            # use the features size to limit number of features samples
            if features_size < 1:
                features_size = int(features_size*len(df_selected['value'].values[-1]))
            df_selected = df_selected.apply(self._df_limit_features, axis=1, args=(features_size, start_pos))
            
        return df_selected