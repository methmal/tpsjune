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





def runtraining(model, seed_arr):
    print(seed_arr)

    test_preds = np.zeros((100000 , 9))
    seed_count = seed_arr.shape[0]
    for ix , seed in enumerate(seed_arr) : 
        utils.set_seed(seed)

        print("Reading data....")
        print(f"MODEL : {model} -----> SEED : {ix} ")
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
        scaler = StandardScaler()
        X = scaler.fit_transform(train)
        X_test = scaler.transform(test)


        clf = model_dispatcher.models[model]
        clf.fit(X,y)
        test_seed_preds = clf.predict_proba(X_test)
        test_preds += test_seed_preds

    test_preds /= seed_count


    print("saving submision predictions")
    sub = pd.read_csv(config.SUBMISION_FILE)
    columns = sub.columns

    print("Saving submision file")

    sub_df = pd.DataFrame(np.column_stack((test_ids , test_preds)) , columns=columns)
    sub_df['id'] = sub_df['id'].astype(int)
    sub_df.to_csv(f'../seed_predictions/submision_{model}_{seed_count}.csv', index=False)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model" ,
            type=str
            )
    args = parser.parse_args()
    runtraining(
            model = args.model,
            seed_arr= np.random.randint(1000, 2000 , size=100 )
            )
