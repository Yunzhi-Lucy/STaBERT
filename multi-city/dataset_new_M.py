import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class HuMobDatasetTask1Train(Dataset):
    def __init__(self):

        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.poi_categories_array = []  
        self.poi_counts_array = []  
        self.city_array = []

        self.df = pd.read_csv('cityB-dataset.csv')
        self.poi_data = pd.read_csv('POIdata_cityB.csv') 
        self.poi_dict = self._process_poi_data()  

        for uid, traj in tqdm(self.df.groupby('uid')):
        #original
            if uid >= 22000:
                traj = traj[traj['d'] < 60]
        #Evaluation experiment(train:uid 0~39999,test:uid 40000~50000)
#            if uid >= 50000:
#                continue  
#            elif 40000 <= uid < 50000:
#                traj = traj[traj['d'] < 60]  

            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                   (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            city = np.full(len(d), 2, dtype=int)

            
            poi_categories = []
            poi_counts = []
            for x, y in zip(input_x, input_y):
                poi_info = self.poi_dict.get((x, y), [(0, 0)])  
                categories, counts = zip(*poi_info) 
                poi_categories.append(list(categories))
                poi_counts.append(list(counts))

            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.poi_categories_array.append(poi_categories)
            self.poi_counts_array.append(poi_counts)
            self.city_array.append(city)
            self.len_array.append(len(d))

        self.df = pd.read_csv('cityC-dataset.csv')
        self.poi_data = pd.read_csv('POIdata_cityC.csv')  
        self.poi_dict = self._process_poi_data()  

        for uid, traj in tqdm(self.df.groupby('uid')):
        #original
            if uid >= 17000:
                traj = traj[traj['d'] < 60]
        #Evaluation experiment(train:uid 0~39999,test:uid 40000~50000)
#            if uid >= 50000:
#                continue  
#            elif 40000 <= uid < 50000:
#                traj = traj[traj['d'] < 60]  

            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                   (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            city = np.full(len(d), 3, dtype=int)

            
            poi_categories = []
            poi_counts = []
            for x, y in zip(input_x, input_y):
                poi_info = self.poi_dict.get((x, y), [(0, 0)])  
                categories, counts = zip(*poi_info)
                poi_categories.append(list(categories))
                poi_counts.append(list(counts))

            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.poi_categories_array.append(poi_categories)
            self.poi_counts_array.append(poi_counts)
            self.city_array.append(city)
            self.len_array.append(len(d))

        self.df = pd.read_csv('cityD-dataset.csv')
        self.poi_data = pd.read_csv('POIdata_cityD.csv')  
        self.poi_dict = self._process_poi_data()  

        for uid, traj in tqdm(self.df.groupby('uid')):
        #original
            if uid >= 3000:
                traj = traj[traj['d'] < 60]
        #Evaluation experiment(train:uid 0~39999,test:uid 40000~50000)
#            if uid >= 50000:
#                continue  
#            elif 40000 <= uid < 50000:
#                traj = traj[traj['d'] < 60]  

            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                   (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            city = np.full(len(d), 4, dtype=int)

            
            poi_categories = []
            poi_counts = []
            for x, y in zip(input_x, input_y):
                poi_info = self.poi_dict.get((x, y), [(0, 0)])  
                categories, counts = zip(*poi_info)  
                poi_categories.append(list(categories))
                poi_counts.append(list(counts))

            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.poi_categories_array.append(poi_categories)
            self.poi_counts_array.append(poi_counts)
            self.city_array.append(city)
            self.len_array.append(len(d))

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def _process_poi_data(self):

        poi_dict = {}
        for _, row in self.poi_data.iterrows():
            key = (row['x'], row['y'])
            if key not in poi_dict:
                poi_dict[key] = []
            poi_dict[key].append((row['category'], row['POI_count']))
        return poi_dict

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        length = torch.tensor(self.len_array[index])
        city = torch.tensor(self.city_array[index])


        poi_categories = self.poi_categories_array[index]
        poi_counts = self.poi_counts_array[index]


        max_poi_len = 85
        poi_category_tensor = torch.zeros(len(poi_categories), max_poi_len, dtype=torch.long)
        poi_count_tensor = torch.zeros(len(poi_counts), max_poi_len, dtype=torch.long)

        for i in range(len(poi_categories)):
            category_len = len(poi_categories[i])
            poi_category_tensor[i, :category_len] = torch.tensor(poi_categories[i], dtype=torch.long)
            poi_count_tensor[i, :category_len] = torch.tensor(poi_counts[i], dtype=torch.long)

        

        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'len': length,
            'poi_categories': poi_category_tensor,  
            'poi_counts': poi_count_tensor, 
            'city': city
        }



class HuMobDatasetTask1Val(Dataset):
    def __init__(self):
        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.city_array = []
        self.poi_categories_array = []  
        self.poi_counts_array = []  

        self.df = pd.read_csv('cityB-dataset.csv')
        self.poi_data = pd.read_csv('POIdata_cityB.csv')  
        self.poi_dict = self._process_poi_data()  

#        original
        self.df = self.df[self.df['uid'] >= 22000]
#        self.df = self.df[(self.df['uid'] >= 40000) & (self.df['uid'] < 50000)]
        for uid, traj in tqdm(self.df.groupby('uid')):
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            city = np.full(len(d), 2, dtype=int)
            


            poi_categories = []
            poi_counts = []
            for x, y in zip(input_x, input_y):
                poi_info = self.poi_dict.get((x, y), [(0, 0)])  
                categories, counts = zip(*poi_info)  
                poi_categories.append(list(categories))
                poi_counts.append(list(counts))

            mask_d_start = 60
            mask_d_end = 74
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.city_array.append(city)
            self.len_array.append(len(d))
            self.poi_categories_array.append(poi_categories)
            self.poi_counts_array.append(poi_counts)

        self.df = pd.read_csv('cityC-dataset.csv')
        self.poi_data = pd.read_csv('POIdata_cityC.csv')  
        self.poi_dict = self._process_poi_data() 

#        original
        self.df = self.df[self.df['uid'] >= 17000]
#        self.df = self.df[(self.df['uid'] >= 40000) & (self.df['uid'] < 50000)]
        for uid, traj in tqdm(self.df.groupby('uid')):
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            city = np.full(len(d), 3, dtype=int)
            


            poi_categories = []
            poi_counts = []
            for x, y in zip(input_x, input_y):
                poi_info = self.poi_dict.get((x, y), [(0, 0)])  
                categories, counts = zip(*poi_info) 
                poi_categories.append(list(categories))
                poi_counts.append(list(counts))

            mask_d_start = 60
            mask_d_end = 74
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.city_array.append(city)
            self.len_array.append(len(d))
            self.poi_categories_array.append(poi_categories)
            self.poi_counts_array.append(poi_counts)

        self.df = pd.read_csv('cityD-dataset.csv')
        self.poi_data = pd.read_csv('POIdata_cityD.csv')  
        self.poi_dict = self._process_poi_data()  

#        original
        self.df = self.df[self.df['uid'] >= 3000]
#        self.df = self.df[(self.df['uid'] >= 40000) & (self.df['uid'] < 50000)]
        for uid, traj in tqdm(self.df.groupby('uid')):
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            city = np.full(len(d), 4, dtype=int)
            


            poi_categories = []
            poi_counts = []
            for x, y in zip(input_x, input_y):
                poi_info = self.poi_dict.get((x, y), [(0, 0)])  
                categories, counts = zip(*poi_info)  
                poi_categories.append(list(categories))
                poi_counts.append(list(counts))

            mask_d_start = 60
            mask_d_end = 74
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.city_array.append(city)
            self.len_array.append(len(d))
            self.poi_categories_array.append(poi_categories)
            self.poi_counts_array.append(poi_counts)

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def _process_poi_data(self):

        poi_dict = {}
        for _, row in self.poi_data.iterrows():
            key = (row['x'], row['y'])
            if key not in poi_dict:
                poi_dict[key] = []
            poi_dict[key].append((row['category'], row['POI_count']))
        return poi_dict

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        city = torch.tensor(self.city_array[index])
        length = torch.tensor(self.len_array[index])


        poi_categories = self.poi_categories_array[index]
        poi_counts = self.poi_counts_array[index]
        

        max_poi_len = 85
        poi_category_tensor = torch.zeros(len(poi_categories), max_poi_len, dtype=torch.long)
        poi_count_tensor = torch.zeros(len(poi_counts), max_poi_len, dtype=torch.long)

        for i in range(len(poi_categories)):
            category_len = len(poi_categories[i])
            poi_category_tensor[i, :category_len] = torch.tensor(poi_categories[i], dtype=torch.long)
            poi_count_tensor[i, :category_len] = torch.tensor(poi_counts[i], dtype=torch.long)

        

        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'city': city,
            'len': length,
            'poi_categories': poi_category_tensor,  
            'poi_counts': poi_count_tensor  
        }

