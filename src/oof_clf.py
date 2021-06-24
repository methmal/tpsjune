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
import os
import shutil
import config 
from sklearn.model_selection import StratifiedKFold

#https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/80809

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = f"{script_name}__folds{config.NFOLD}"
print(f"Model: {MODEL_NAME}")





def runtraining():
    utils.set_seed(config.RANDOM_SEED)

    print("Reading data....")
    train = pd.read_csv(config.TRAIN_FILE)
    train = utils.target_encode(train)
    test = pd.read_csv(config.TEST_FILE)


    y = train.target.values
    train_ids = train.id.values.astype(int)

    train = train.drop(['id' , 'target'] , axis=1)
    feature_list = train.columns

    test_ids = test.id.values.astype(int)
    test = test[feature_list]
    
    #feature engineering 
    #noramalized feature_
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(train)
    # X_test = scaler.fit_transform(test)
    X = train.values.astype(float)
    X_test = test.values.astype(float)

    clf = []
    folds = StratifiedKFold(n_splits=config.NFOLD , shuffle=True , random_state=config.RANDOM_SEED)
    oof_preds = np.zeros((len(train) , 9))
    test_preds = np.zeros((len(test) , 9))
    comulative_log_loss =0 
    for fold_ , (trn_ , val_) in enumerate(folds.split(X,y)):
        print(f"")
        trn_x , trn_y = X[trn_ , : ] , y[trn_]
        val_x , val_y = X[val_ , :] , y[val_]

        #feature scaling
        scaler = StandardScaler()
        fitscaler = scaler.fit(trn_x)

        trn_x = fitscaler.transform(trn_x)
        val_x = fitscaler.transform(val_x)
        X_test = fitscaler.transform(X_test)




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
        model.fit(trn_x,trn_y,eval_set=[(val_x,val_y)],early_stopping_rounds=100,verbose=False)
        val_pred = model.predict_proba(val_x)
        test_fold_preds = model.predict_proba(X_test)
        fold_log_loss = log_loss(val_y, val_pred)
        comulative_log_loss += fold_log_loss
        avg_log_loss = comulative_log_loss / (fold_ + 1)
        print(f"Current fold: {fold_}      LOG_LOSS= {fold_log_loss}       AVERAGE_LOG_LOSS = {avg_log_loss}")

        oof_preds[val_ , :] = val_pred
        test_preds += test_fold_preds

    test_preds /= config.NFOLD

    OOF_log_loss = log_loss(y , oof_preds)
    print(f"OOF LOG_LOSS = {OOF_log_loss}")

    print("saving OOF predictions")
    sub = pd.read_csv(config.SUBMISION_FILE)
    columns = sub.columns
    oof_preds = pd.DataFrame(np.column_stack((train_ids , oof_preds)) , columns=columns)
    oof_preds['id'] = oof_preds['id'].astype(int)

    oof_preds.to_csv(f'../kfold/{MODEL_NAME}__{OOF_log_loss}.csv', index=False)

    print("Saving submision file")

    sub_df = pd.DataFrame(np.column_stack((test_ids , test_preds)) , columns=columns)
    sub_df['id'] = sub_df['id'].astype(int)
    sub_df.to_csv(f'../model_predictions/submision_{MODEL_NAME}_{OOF_log_loss}.csv', index=False)


if __name__ == "__main__" :
    runtraining()
    # df_sub = pd.read_csv("../model_predictions/submision_rf_clf__folds5_1.7447258575478999.csv")
    # print(df_sub.dtypes)
