from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pystacknet.pystacknet import StackNetClassifier
import sys
sys.modules['sklearn.externals.joblib'] = joblib

cb_params = {'iterations':15046,
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

rf_params = {
	'n_estimators': 999,
	'max_depth': 39,
	'min_samples_split': 19,
        'min_samples_leaf': 2,
    }

xb_params = {
                 'learning_rate': 0.032987265583032645 , 
                 'gamma': 0.2516706421997758 ,
                 'reg_alpha': 3 , 
                 'reg_lambda': 5 ,
                 'n_estimators': 492, 
                 'colsample_bynode': 0.3887329289320089,
                 'colsample_bylevel': 0.7358670152042927,
                 'subsample':0.7334296724195702,
                 'min_child_weight': 129,
                 'colsample_bytree': 0.26140522927698306, 

    }

lgbm_params ={
		'objective': 'multiclass', 
		'num_class': 9, 
                'metric': 'multi_logloss', 
                'verbosity': -1, 
                'boosting_type': 'gbdt', 
                'learning_rate': 0.03, 
                'random_state': 42, 
                'feature_pre_filter': False, 
                'lambda_l1': 1.2301296575987508e-07, 
                'lambda_l2': 6.513713579597348, 
                'num_leaves': 10, 
                'feature_fraction': 0.4, 
                'bagging_fraction': 1.0, 
                'bagging_freq': 0, 
                'min_child_samples': 10 ,
            }
pystack_params = [
        [
        CatBoostClassifier(**cb_params),
        RandomForestClassifier(**rf_params),
        XGBClassifier( objective='multi:softprob', eval_metric = "mlogloss", num_class = 9, tree_method = 'gpu_hist', max_depth = 13, use_label_encoder=False, **xb_params),
        LGBMClassifier(**lgbm_params)
            ],
        [
        RandomForestClassifier(**rf_params),
            ],
        
        ]
 
pystack_lte_params = [
        [
        CatBoostClassifier(**cb_params),
            ],
        [
        XGBClassifier( objective='multi:softprob', eval_metric = "mlogloss", num_class = 9, tree_method = 'gpu_hist', max_depth = 13, use_label_encoder=False, **xb_params),
            ],
        
        ]
 




models = {
        "cb" : CatBoostClassifier(**cb_params),
        "rf" : RandomForestClassifier(**rf_params),
        "xb" : XGBClassifier( objective='multi:softprob', eval_metric = "mlogloss", num_class = 9, tree_method = 'gpu_hist', max_depth = 13, use_label_encoder=False, **xb_params),
        "lgb": LGBMClassifier(**lgbm_params),
        "stacknet" : StackNetClassifier(pystack_params, metric="logloss", folds=5,
                restacking=False,use_retraining=True, use_proba=True, 
                random_state=42,n_jobs=1, verbose=1) ,
        "stacknetlte" : StackNetClassifier(pystack_lte_params, metric="logloss", folds=5,
                restacking=False,use_retraining=True, use_proba=True, 
                random_state=42,n_jobs=1, verbose=1)
}
