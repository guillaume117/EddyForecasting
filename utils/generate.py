import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
import netCDF4
import h5netcdf
from datetime import datetime
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from utils.util import addGitignore
import os




class GenerateDataset():
    """ 
    This class is used to generate the dataset from the OSSE and eddies files.
    It is also used to split the dataset into train and validation datasets, and to generate the mask 
    for the label and input datasets (we don't want to train our model over land).

    """


    def __init__(self):
        self.OSSE_test  = None
        self.OSSE_train = None
        self.eddies_train = None
        self.following_dates = []
        self.train_path = None
        self.val_path = None
        self.nan_mask_label_deconv = None
        self.nan_mask_label = None
        self.generate()


    def generate(self):
        #files = []
        #for dirname, _, filenames in os.walk('./data/'):
        #    for filename in filenames:
        #        print(os.path.join(dirname, filename))
        #        files.append(os.path.join(dirname, filename))
        #self.eddies_train = xr.open_dataset(files[1])
        #self.OSSE_test = xr.open_dataset(files[0])
        #self.OSSE_train = xr.open_dataset(files[2])
        #self.OSSE_train = self.OSSE_train.rename({"time_counter":"time"})
        self.eddies_train = xr.open_dataset('data/eddies_TRAIN_for_FORECAST.nc')
        self.OSSE_test = xr.open_dataset('data/OSSE_U_V_SLA_SST_2015_TEST_for_FORECAST.nc')
        self.OSSE_train = xr.open_dataset('data/OSSE_U_V_SLA_SST_2015_TRAIN_for_FORECAST.nc')
        self.OSSE_train = self.OSSE_train.rename({"time_counter":"time"})
       

        path = 'dataset'
        addGitignore(path)
    
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        pass



    def followingDates(self,num_date,type='Train'):
        """ This function is used to get the following dates index for the train and test datasets.
        It is used to generate the dataset, in fact we need to have a sequence of num_date dates to predict the next num_date dates.
        input : num_date : int : the number of dates to predict
                type : str : 'Train' or 'Test'
        output : followingDatesIndex : list : the list of the index of the following dates
        """
        if (type =='Train'):
            print('train')
            dates_list = [datetime.strptime(str(self.OSSE_train.time[i].values)[:10], '%Y-%m-%d') for i in range(len(self.OSSE_train.time))]
            followingDatesIndex = np.where([all((dates_list[j+i+1] - dates_list[j+i]).days==1 for i in range(num_date)) for j in range(len(dates_list) - 2*num_date)])[0]

        if (type =='Test'):
            followingDatesIndex=[0,10,20,30]

        return followingDatesIndex
    
    def generateAndSplitDataset(self,X,y, validation_fraction=0.2,running_instance = 'Running_instance'):
        """ This function is used to generate and split the dataset into train and validation datasets.
        input : X : tensor : the input dataset
                y : tensor : the target dataset
                validation_fraction : float : the fraction of the dataset to use as validation dataset
                running_instance : str : the name of the running instance
        output : train_dataset : tensor : the train dataset
                 val_dataset : tensor : the validation dataset
        """

        
    
        dataset = TensorDataset(X, y)
        validation_fraction = 0.2
        val_size = int(validation_fraction * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        torch.save(train_dataset,self.train_path)
        torch.save(val_dataset,self.val_path)
        torch.save(self.nan_mask_label_deconv,self.nan_mask_label_deconv_path)
        torch.save(self.nan_mask_label,self.nan_mask_label_path)
        
        return train_dataset,val_dataset
    
    
    def processingData(self,num_date,type ='Train',running_instance = 'Train_2',generate =False, validation_fraction=0.2):
        """ This function is used to preprocess the data, it generates the dataset and split it into train and validation datasets.
        input : num_date : int : the number of dates to predict
                type : str : 'Train' or 'Test'
                running_instance : str : the name of the running instance
                generate : bool : whether to generate the dataset or not
                validation_fraction : float : the fraction of the dataset to use as validation dataset
        output : train_dataset : tensor : the train dataset
                 val_dataset : tensor : the validation dataset
                 nan_mask_label_deconv : tensor : the mask for the label dataset
                 nan_mask_label : tensor : the mask for the input dataset
        """
        path = 'dataset/'+running_instance
        addGitignore(path)

     
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        pass
        self.train_path = f'{path}/TrainDataset_{running_instance}__dataset.pt'
        self.val_path = f'{path}/ValDataset_{running_instance}__dataset.pt'
        self.nan_mask_label_deconv_path = f'{path}/nan_mask_label_deconv_{running_instance}.pt'
        self.nan_mask_label_path = f'{path}/nan_mask_label_{running_instance}.pt'

        followingDatesIndex= self.followingDates(num_date,type=type)
        if type =='Train':
            X_dataset_train= torch.tensor(np.array([[self.OSSE_train.sossheig.values[i+j]for i in range(num_date)] for j in followingDatesIndex]))
            y_dataset_train = torch.tensor(np.array([[self.eddies_train.eddies.values[i+j]for i in range(num_date,2*num_date)]for j in followingDatesIndex]))
            trans = transforms.Resize((96*4,192*4))
            self.nan_mask_label_deconv = torch.isnan(y_dataset_train)
            X_dataset_train=trans(X_dataset_train)
            y_dataset_train=trans(y_dataset_train)
            #Separation of eddies results on 3 layers

            condition_0 = (y_dataset_train<=0.5)
            condition_1 = (y_dataset_train>0.5)&(y_dataset_train<1.5)
            condition_2 = (y_dataset_train>=1.5)
            y_dataset_train= torch.cat([torch.where(condition_0,torch.tensor(1),torch.tensor(0)),
                             torch.where(condition_1,torch.tensor(1),torch.tensor(0)),torch.where(condition_2,torch.tensor(1),torch.tensor(0))],dim=1)
            self.nan_mask_label = torch.isnan(y_dataset_train)
            y_dataset_train = torch.where(self.nan_mask_label, torch.tensor(0), y_dataset_train)
            nan_mask = torch.isnan(X_dataset_train)
            X_dataset_train = torch.where(nan_mask, torch.tensor(0.0), X_dataset_train)
            print("*"*100)
            print(f"Dataset has been initialised, inputs shape {X_dataset_train.shape}, {len(X_dataset_train)} elements of depth {len(X_dataset_train[0])}.")
            print("Use generateAndSplit to generate and save TRAIN and VAL")
            print(f"Target shape is a tensor of depth {len(y_dataset_train[0])}")
            print("*"*100)
            if generate ==True :
                train_dataset,val_dataset = self.generateAndSplitDataset(X_dataset_train,y_dataset_train,validation_fraction=validation_fraction,running_instance=running_instance)
                del X_dataset_train,y_dataset_train
                print("*"*100)
                print(f"train_dataset has been created, inputs shape, {len(train_dataset)} elements.")
                print(f"recorded at {self.train_path}")
                print(f"train_dataset has been created, inputs shape, {len(val_dataset)} elements.")
                print(f"recorded at {self.val_path}")
                print(f"nan_mask_label_deconv has been created shape, {self.nan_mask_label_deconv.shape}")
                print(f"recorded at {self.nan_mask_label_deconv_path}")
                print(f"nan_mask_label has been created shape, {self.nan_mask_label.shape}")
                print(f"recorded at {self.nan_mask_label_path}")
                
                print("*"*100)
             
            else :
                pass
        
        if type == 'Test':
            print('test')
            X_dataset_test = torch.tensor(np.array([[self.OSSE_test.sossheig.values[i+j]for i in range(num_date)] for j in followingDatesIndex]))
            
            return X_dataset_test
    
    def getTrainPath(self):
        return self.train_path
    
    def getValPath(self):    
        return self.val_path
    
    def getNanMaskLabelDeconvPath(self):
        return self.nan_mask_label_deconv_path
    
    def getNanMaskLabelPath(self):
        return self.nan_mask_label_path
    
        
    
