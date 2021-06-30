import pandas as pd
import numpy as np
import torch
import config

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import category_encoders as ce
from dataset import TabDataset
from  model import Model
from engine import Engine
import sys
from multimodel import MulticlassClassification
import torch.nn as nn
from catboost import CatBoostClassifier
import optuna
import optuna.integration.lightgbm as lgb
from lightgbm import LGBMClassifier
from optuna.integration import LightGBMPruningCallback
from functools import partial



def target_encode(df):
    class2idx = {
            "Class_1" : 0 ,
            "Class_2" : 1 ,
            "Class_3" : 2 ,
            "Class_4" : 3 ,
            "Class_5" : 4 ,
            "Class_6" : 5 ,
            "Class_7" : 6 ,
            "Class_8" : 7 ,
            "Class_9" : 8 ,
            }
    idx2class = {v : k for k ,v in class2idx.items()}
    df["target"].replace(class2idx , inplace=True)
    return df





def runtraining():

    train = pd.read_csv(config.TRAIN_FILE)
    train = target_encode(train)
    X = train.drop(['id' , 'target'] , axis=1)
    y= train.target

    #tune_LightGBMTunerCV
    dtrain = lgb.Dataset(X, label=y)

    params = {
        "objective": "multiclass",
        "num_class": 9,
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        'learning_rate': 0.03,
        'random_state': 42 ,
    }
    tuner = lgb.LightGBMTunerCV(
        params, dtrain, verbose_eval=150, early_stopping_rounds=100, 
        folds=StratifiedKFold(n_splits=5, shuffle=True)
    )
    tuner.run()
    print(tuner.best_params)
    print(tuner.best_score)
	
    

if __name__ == "__main__" :
    runtraining()
