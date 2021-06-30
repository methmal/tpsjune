import numpy as np
import random
import torch
import os
def set_seed(seed=0):
    np.random.seed=seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
