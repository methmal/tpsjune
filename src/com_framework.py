import os
import shutil
import pandas as pd
import numpy as np
import config

#taken form this kaggle notbook
#https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/80809

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = f"{script_name}__folds{config.NFOLD}"

print(f'Model: {MODEL_NAME}')

