from Utils.classes import DataSet, Model
from Utils.utils import auc_score, acc_score, precision_at_k, mpr,  hit_rate_at_k
import time
from Resources import config




wall_start = time.time()
sample_method = config.method
d = DataSet(**config.data_configs)


if sample_method =='random':
    bpr_params = config.bpr_params_random
else:
    bpr_params = config.bpr_params_pop

bpr_params['n_users'] = d.users
bpr_params['n_items'] = d.items


bpr = Model(**bpr_params)

if config.tune:
    tuning_results = bpr.hyperparmas_tune(train=d.train, validation=d.validation, epochs=200, max_evals=100, sample_method=sample_method) #this could take long
    for k in tuning_results.keys(): #update with optimized values
        bpr_params[k] = tuning_results[k]

    print('Best dict for popularity', bpr_params)


if config.EVALUATE:
    bpr.fit(train=d.train,  epochs=config.epochs, batch_size=config.batch_size,  sample_method=sample_method)


    preds=bpr.predict_all(use_bias=True)


    print('ACC score Validation set: ',acc_score(preds, d.validation, sample_method))
    print('AUC score Validation set: ',auc_score(preds, d.validation))
    if config.data_configs['validation_percent'] is not None: #more than one item per user in validation
        print('Precision at K Validation Set: ',precision_at_k(preds, d.validation))
    else:
        print('Hit rate at K Validation Set: ',hit_rate_at_k(preds, d.validation))
    print('MPR Validation Set: ', mpr(preds, d.validation, d.unranked_items_per_user))


if config.SAVE_RESULTS:
    # train on all of the data (no split to validation)
    d = DataSet(validation_percent=0)
    bpr.fit(train=d.train, epochs=config.epochs, batch_size=config.batch_size, sample_method=sample_method)

    if sample_method=='random':
        test_set=d.random_test.copy()
        final_result=d.random_test.copy()
    else:
        test_set=d.popularity_test.copy()
        final_result=d.popularity_test.copy()

    #encode the users and items like we did in the CSR
    test_set['UserID']=test_set['UserID'].astype('category').cat.codes
    test_set['Item1']=test_set['Item1'].astype('category').cat.codes
    test_set['Item2']=test_set['Item2'].astype('category').cat.codes
    test_set['bitClassification']=test_set.apply(lambda x: bpr.predict(x[0], x[1], x[2], use_bias=True), axis=1)
    # in the assignment it says that 0 if you predict that the first item was the item that was liked by the user and vice versa
    test_set['bitClassification']=1-test_set['bitClassification']
    # since we encoded the users and the items lets copy our prediction to the final test set
    final_result['bitClassification']= test_set['bitClassification']
    #save to local fs
    final_result.to_csv('Resources/{}_201641164_037098985_205766496.csv'.format(sample_method), index=False)

wall_end = time.time()
print('Wall time in seconds: ', round(wall_end - wall_start,1))






