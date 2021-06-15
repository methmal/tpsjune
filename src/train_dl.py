import pandas as pd
import torch
import config
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import numpy as np
from dataset import TabDataset
from  model import Model
from engine import Engine
import sys
from multimodel import MulticlassClassification
import torch.nn as nn

def get_class_distrbution(obj):
    count_dic = {
            "Class_1" : 0 ,
            "Class_2" : 0 ,
            "Class_3" : 0 ,
            "Class_4" : 0 ,
            "Class_5" : 0 ,
            "Class_6" : 0 ,
            "Class_7" : 0 ,
            "Class_8" : 0 ,
            "Class_9" : 0 ,
            }
    for i in obj:
        if i == 0 :
            count_dic['Class_1'] +=1
        elif i == 1 :
            count_dic['Class_2'] +=1
        elif i == 2 :
            count_dic['Class_3'] +=1
        elif i == 3 :
            count_dic['Class_4'] +=1
        elif i == 4 :
            count_dic['Class_5'] +=1
        elif i == 5 :
            count_dic['Class_6'] +=1
        elif i == 6 :
            count_dic['Class_7'] +=1
        elif i == 7 :
            count_dic['Class_8'] +=1
        elif i == 8 :
            count_dic['Class_9'] +=1
        else: 
            print("check classes")
    return count_dic


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

def runtraining(fold ,save_model=False):
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
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    ytrain = train_df[target_columns].values
    xvalid = scaler.transform(xvalid)
    yvalid = valid_df[target_columns].values


    train_dataset = TabDataset(xtrain , ytrain)
    valid_dataset = TabDataset(xvalid, yvalid)


    #calculate weighted sampler
    temp_target_list = [] 
    for item in train_dataset :
         temp_target_list.append(item['y'].tolist())
    target_list = [item for sublist in temp_target_list  for item in sublist]
    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]
    class_count = [i for i in get_class_distrbution(ytrain).values()]
    class_weights = 1./torch.tensor(class_count , dtype=torch.float)
    class_weights_all = class_weights[target_list]

    weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
            )
    train_loader = torch.utils.data.DataLoader(
            train_dataset , batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS ,sampler=weighted_sampler
            )
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.VALID_BATCH_SIZE , num_workers=config.NUM_WORKERS
            )

    # model = Model(
    #         nfeatures=xtrain.shape[1],
    #         ntargets=ytrain.shape[1],
    #         nlayers=2,
    #         hidden_size=124,
    #         dropout=0.3
    #         )
    model = MulticlassClassification(
            num_feature=xtrain.shape[1],
            num_class=9
            )
    model.to(config.DEVICE)
    optimzer = torch.optim.Adam(model.parameters() , config.LERNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    eng = Engine(model , optimzer , criterion, config.DEVICE )
    best_loss = np.inf
    early_stoping_itr = 10
    early_stoping_count = 0

    for epoch in range(config.EPOCH):
        train_loss , train_acc = eng.train_fn(train_loader)
        valid_loss ,valid_acc =  eng.eval_fn(valid_loader)
        print(f"{fold}----> train_loss : {train_loss} :: valid_loss : {valid_loss} :: train_acc : {train_acc} ::: valid_acc : { valid_acc }") 
        if valid_loss < best_loss: 
            best_loss = valid_loss
            if save_model :
                torch.save(torch.save(model.save_dict() , f"model_{fold}.bin"))
        else :
            early_stoping_count +=1
        if early_stoping_count > early_stoping_itr:
            break

if __name__ == "__main__" :
    runtraining(0)
    # runtraining(1)
    # runtraining(2)
    # runtraining(3)
    # runtraining(4)
