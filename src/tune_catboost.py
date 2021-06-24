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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import optuna
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




def objective(trial):
    
    params = {'iterations':trial.suggest_int("iterations", 4000, 25000),
              'od_wait':trial.suggest_int('od_wait', 500, 2300),
             'loss_function':'MultiClass',
              'task_type':"GPU",
              'eval_metric':'MultiClass',
              'leaf_estimation_method':'Newton',
              'bootstrap_type': 'Bernoulli',
              'learning_rate' : trial.suggest_uniform('learning_rate',0.02,1),
              'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
              'subsample': trial.suggest_uniform('subsample',0,1),
              'random_strength': trial.suggest_uniform('random_strength',10,50),
              'depth': trial.suggest_int('depth',1,15),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
              'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
               }

    all_loss = []
    for fol_ in range(5):
        temp_loss = runtraining(fol_ , params , save_model=False)
        all_loss.append(temp_loss)

    return np.mean(all_loss)

def runtraining(fold,params=None ,save_model=False):

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
    
    model = CatBoostClassifier(**params)  
    model.fit(xtrain,ytrain,eval_set=[(xvalid,yvalid)],early_stopping_rounds=100,verbose=False)
    y_preds = model.predict_proba(xvalid)
    log_loss_multi = log_loss(yvalid, y_preds)
    return log_loss_multi


if __name__ == "__main__" :
    partial_obj= partial(objective)
    study = optuna.create_study(direction='minimize')
    study.optimize(partial_obj, n_trials=150)
    print('Number of finished trials:', len(study.trials))
    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
