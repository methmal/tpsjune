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

def runtraining(fold,params=None ,save_model=False):
    utils.set_seed(config.RANDOM_SEED)

    df = pd.read_csv(config.TRAIN_FOLD)
    df_test = pd.read_csv(config.TEST_FILE)
    df = df.drop(['id'] , axis=1)
    df_test = df_test.drop(['id'] , axis=1)
    df = target_encode(df)

    feature_columns = [f'feature_{i}' for i in range(0,75)]
    target_columns = ['target']

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    test_df = df_test


    xtrain = train_df[feature_columns]
    xvalid = valid_df[feature_columns]
    xtest = test_df
    #noramalized feature_
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    ytrain = train_df[target_columns].values
    xvalid = scaler.transform(xvalid)
    yvalid = valid_df[target_columns].values

    xtest = scaler.fit_transform(xtest)
    params = {'iterations':15046,
            'od_wait': 913 ,
             'loss_function':'MultiClass',
              'task_type':"GPU",
              'eval_metric':'MultiClass',
              'leaf_estimation_method':'Newton',
              'bootstrap_type': 'Bernoulli',
              'learning_rate' : 0.0211020670690157 ,
              'reg_lambda': 54.82598802626106 ,
              'subsample': 0.7353916510620078 ,
              'random_strength': 35.02506949376338 ,
              'depth': 9,
              'min_data_in_leaf': 25 ,
              'leaf_estimation_iterations': 1 ,
               }

    
    model = CatBoostClassifier(**params)  
    model.fit(xtrain,ytrain,eval_set=[(xvalid,yvalid)],early_stopping_rounds=100,verbose=False)
    y_preds = model.predict_proba(xvalid)
    test_fold_preds = model.predict_proba(xtest)
    log_loss_multi = log_loss(yvalid, y_preds)
    return log_loss_multi , y_preds , yvalid , test_fold_preds


if __name__ == "__main__" :
    # fold_log_loss , fold_ypreds , fold_yvalid , fold_test_preds = runtraining(0)
    # loss_0 , _ , _ , fold_test_preds_0 = runtraining(0)
    # loss_1 , _ , _ , fold_test_preds_1 = runtraining(1)
    # loss_2 , _ , _ , fold_test_preds_2 = runtraining(2)
    # loss_3 , _ , _ , fold_test_preds_3 = runtraining(3)
    # loss_4 , _ , _ , fold_test_preds_4 = runtraining(4)
    # loss_arr = np.array([loss_0, loss_1 , loss_2 , loss_3 , loss_4])
    # print(loss_arr)
    # loss_avg =np.mean(loss_arr)
    # print(f"loss_avg--->{loss_avg}")

    # fold_test_preds = (fold_test_preds_0 + fold_test_preds_1 + fold_test_preds_2 + fold_test_preds_3 + fold_test_preds_4) / 5
    # print(fold_test_preds.shape)
    # sample_submission = pd.read_csv("../input/sample_submission.csv")
    # classes = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
    # sample_submission.drop(columns=classes , inplace=True)
    # # submision = sample_submission.join(pd.DataFrame(data=fold_test_preds) ,columns=classes)
    # submision = (sample_submission.join(pd.DataFrame(data=fold_test_preds, 
    #                                         columns=classes)))
    # submision.to_csv("submision.csv", index=False)

    df = pd.read_csv(config.TRAIN_FOLD)
    df_test = pd.read_csv(config.TEST_FILE)
    print((df.shape)[0])
    print((df_test.shape)[0])
    len_train = (df.shape)[0]
    len_test = (df_test.shape)[0]
    
    oof_logloss = 0
    oof_preds = np.zeros((len_train,1))
    test_preds = np.zeros((len_test,1))
