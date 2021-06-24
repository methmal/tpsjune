import pandas as pd
import torch
import config
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import numpy as np
from dataset import TabDataset
from  model import Model
from engine import Engine
import sys
from multimodel import MulticlassClassification
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import optuna
from functools import partial
import utils



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




def objective(trial):
    utils.set_seed(config.RANDOM_SEED)
    
    params = {
	'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
	'max_depth': trial.suggest_int('max_depth', 4, 50),
	'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
    }
    all_loss = []
    for fol_ in range(5):
        temp_loss = runtraining(fol_ , params , save_model=False)
        all_loss.append(temp_loss)

    return np.mean(all_loss)

def runtraining(fold,params=None ,save_model=False):
    utils.set_seed(config.RANDOM_SEED)

    df = pd.read_csv(config.TRAIN_FOLD)
    df = df.drop(['id'] , axis=1)
    df = target_encode(df)

    feature_columns = [f'feature_{i}' for i in range(0,75)]
    target_columns = ['target']

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)


    xtrain = train_df[feature_columns]
    xvalid = valid_df[feature_columns]
    #noramalized feature_
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    ytrain = train_df[target_columns].values
    xvalid = scaler.transform(xvalid)
    yvalid = valid_df[target_columns].values
    
    model = RandomForestClassifier(random_state=config.RANDOM_SEED , **params)  
    model.fit(xtrain,ytrain.ravel())
    y_preds = model.predict_proba(xvalid)
    log_loss_multi = log_loss(yvalid, y_preds)
    return log_loss_multi


if __name__ == "__main__" :
    utils.set_seed(config.RANDOM_SEED)
    partial_obj= partial(objective)
    study = optuna.create_study(direction='minimize')
    study.optimize(partial_obj, n_trials=150)
    print('Number of finished trials:', len(study.trials))
    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
