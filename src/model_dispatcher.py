from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

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

models = {
        "cb" : CatBoostClassifier(**cb_params),
        "rf" : RandomForestClassifier(**rf_params)
}
