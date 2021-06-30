import pandas as pd
import config
from sklearn import model_selection
from utils import set_seed

if __name__ == "__main__":
    set_seed(config.RANDOM_SEED)
    df = pd.read_csv(config.TRAIN_FILE)
    df.loc[: , 'kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5 , shuffle=True , random_state=42)
    y = df.target.values
    for f_ , (t_ , v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_ , 'kfold'] = f_

    df.to_csv("train_folds.csv" , index=False)



    # df = pd.read_csv(config.TRAIN_FOLD)
    # print(df.head())
    # print(df.columns)
