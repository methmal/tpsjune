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
import argparse
import model_dispatcher

#https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/80809

# script_name = os.path.basename(__file__).split('.')[0]
# MODEL_NAME = f"{script_name}__folds{config.NFOLD}"
# print(f"Model: {MODEL_NAME}")





def runtraining(model):
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
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    # X = train.values.astype(float)
    # X_test = test.values.astype(float)

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
        # scaler = StandardScaler()

        # trn_x = scaler.fit_transform(trn_x)
        # val_x = scaler.transform(val_x)
        # X_test = scaler.transform(X_test)



        clf = model_dispatcher.models[model]
        # model = CatBoostClassifier(**params)  
        if model == "cb":
            clf.fit(trn_x,trn_y,eval_set=[(val_x,val_y)],early_stopping_rounds=100,verbose=False)
        else :
            clf.fit(trn_x,trn_y)
        val_pred = clf.predict_proba(val_x)
        test_fold_preds = clf.predict_proba(X_test)
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

    oof_preds.to_csv(f'../kfold/{model}__{OOF_log_loss}.csv', index=False)

    print("Saving submision file")

    sub_df = pd.DataFrame(np.column_stack((test_ids , test_preds)) , columns=columns)
    sub_df['id'] = sub_df['id'].astype(int)
    sub_df.to_csv(f'../model_predictions/submision_{model}_{OOF_log_loss}.csv', index=False)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model" ,
            type=str
            )
    args = parser.parse_args()
    runtraining(
            model = args.model
            )
