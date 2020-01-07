from Utils.classes import DataSet, Model
from Utils.utils import auc_score, precision_at_k, mpr
import time

tune=False
data_configs = {'seed': 7 , 'validation_percent': 0.2}



wall_start = time.time()
d = DataSet(**data_configs)
data_prep_time=time.time()
print('Time to prep data', data_prep_time- wall_start)


bpr_params = {'latent_dim': 38, 'n_users': d.users, 'n_items': d.items, 'l_users':0.02106964961227692,
              'l_items':0.016332371340206816, 'l_bias_items':0.05722601017641107, 'learning_rate': 0.13460602180353268,
              'use_decay': False , 'learning_decay':0, 'seed': data_configs['seed']}


bpr = Model(**bpr_params)

if tune:
    tuning_results = bpr.hyperparmas_tune(train=d.train, validation=d.validation, epochs=300, max_evals=100) #this could take long
    for k in tuning_results.keys(): #update with optimized values
        bpr_params[k] = tuning_results[k]


bpr.fit(train=d.train,  epochs=250, batch_size=100,  sample_method='random')
model_train_time=time.time()
print('Time to train model in seconds: ', round(model_train_time-data_prep_time,1))
preds=bpr.predict_all()
predictions_time=time.time()
print('Time to generate all predictions in seconds: ', round(predictions_time-model_train_time,1))

print('AUC score Train set: ', auc_score(preds, d.train))
print('AUC score Validation set: ',auc_score(preds, d.validation))
auc_time = time.time()
print('Time to evaluate auc in seconds: ', round(auc_time-predictions_time,1))

print('Precision at K Train Set: ', precision_at_k(preds, d.train))
print('Precision at K Validation Set: ',precision_at_k(preds, d.validation))
precision_time = time.time()
print('Time to evaluate precision at k in seconds: ', round(precision_time-auc_time,1))

print('MPR Validation Set: ', mpr(preds, d.validation))
mpr_time = time.time()
print('Time to evaluate mpr in seconds: ', round(mpr_time-precision_time,1))
wall_end = time.time()
print('Wall time in seconds: ', round(wall_end - wall_start,1))



