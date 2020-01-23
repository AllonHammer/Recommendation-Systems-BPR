import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import rankdata


def item_popularity(csr):
    """

    :param csr: scipy.sparse.csr_matrix() with shape (m_users, n_items)
    :return: np.array (n_items,) with the probability to sample each itm according to popularity (i.e. non-zero items/positive feedback in the csr matrix)
    """

    items = np.arange(csr.shape[1])
    # sum occurrences of each item
    popularity = csr.sum(axis=0) #(items,)
    # convert to array
    pop_array = np.squeeze(np.asarray(popularity))
    # normalize between 0-1
    proba_lst =  pop_array[items] / pop_array.sum()
    return proba_lst




def sample_batch_uij(batch_size, n_users, n_items, indices, indptr,  proba_lst, sample_by='random'):
    """
    samples a batch of <u,i,j> .  since passing the full CSR matrix has a huge overhead we take only the indices and indptr
    :param batch_size: int the number of <u,i,j> to sample
    :param n_users: int number of users
    :param n_items: int number of items
    :param indices: scipy.sparse.scr_matrix().indices
    :param indptr: scipy.sparse.scr_matrix().indptr
    :param proba_lst: np.array (n_users,), 1-d array, indices are item_ids and values are the probability according to item popularity
    :param sample_by: string 'random/popularity', represents the sampling method

    :return: np.array (batch_size, ) represents batch_u
    :return: np.array (batch_size, ) represents batch_i
    :return: np.array (batch_size, ) represents batch_j


    """

    # In case batch_size is larger than the number of users
    batch_size = min(batch_size, n_users)

    # sample user without return
    batch_u=np.random.choice(np.arange(n_users), size = batch_size, replace = False)
    batch_i=np.zeros(batch_size)
    batch_j=np.zeros(batch_size)
    for idx, u in enumerate(batch_u):
        # get positive (non zero) occurrences from the csr matrix (without passing the full matrix, just pointers)
        positive_items=indices[indptr[u]: indptr[u+1]]
        # sample positive
        i=np.random.choice(positive_items)
        if sample_by != 'popularity': # if we sample by random then we do not need to pass proba_lst
            proba_lst = None

        j = np.random.choice(n_items, p=proba_lst) #sample negative from all items

        while j in positive_items: #if sampled item is a positive item sample again
            j = np.random.choice(n_items, p=proba_lst)

        batch_i[idx]=i
        batch_j[idx]=j

    return batch_u.astype(int), batch_i.astype(int), batch_j.astype(int)


def acc_score(preds, csr, sample_by='random'):
    """
    Create validation set with positive (label=1) and negative (label=0) items, and attach predictions.
    Now, each row is <user, item, label, prediction>
    We calculate the accuracy score for this dataset (note: this dataset  is balanced).
    We also use this evaluation metric for hyperparam tuning, since the test set contains <user, item1, item2>
    and we need to assign 1 or 0 to each item. I assume it is evaluated using accuracy.

    :param preds: np.array (m- number of users, n- number of items) predictions for all users and items
    :param csr: scipy.sparse.csr_matrix() representation of validation set
    :return: float (0..1) acc score
    """

    if sample_by=='popularity':
        proba_list=item_popularity(csr)
    else:
        proba_list=[]

    acc_results=[]
    # since we sample different negatives each time we sample let's do this 10 times and average

    for _ in range(10):
        # Get a full batch of the validation set including negatives
        batch_u, batch_i, bacth_j = sample_batch_uij(batch_size=csr.shape[0],n_users=csr.shape[0], n_items=csr.shape[1],
                                   indices=csr.indices, indptr=csr.indptr, proba_lst=proba_list, sample_by=sample_by)
        val = []
        for sample in zip(batch_u, batch_i, bacth_j):
            u, i, j = sample
            pos = [u, i, 1]  # append positive as 1 label
            neg = [u, j, 0]  # append negative as 0 label
            val.append(pos)
            val.append(neg)
        y_hat  = [preds[sample[0], sample[1]] for sample in val] #get predictions for  each row
        y_pred = [1 if y>=0.5 else 0 for y in y_hat]
        y_true = [sample[2] for sample in val]

        acc = accuracy_score(y_true, y_pred)
        acc_results.append(acc)
    print('ACC Cross Validation results', np.round(acc_results,3))
    acc_results = np.array(acc_results).mean()
    return np.round(acc_results,3)


def auc_score(preds, csr):
    """
    compute auc score per user, and average across all users

    :param preds: np.array (m- number of users, n- number of items) predictions for all users and items
    :param csr: scipy.sparse.csr_matrix() representation of validation set
    :return: float (0..1) auc score
    """

    auc = 0.0
    for user in range(csr.shape[0]):
        positives = csr[user].indices
        y_true = np.zeros(csr.shape[1])
        y_true[positives] = 1
        y_pred = preds[user]
        auc += roc_auc_score(y_true, y_pred)
    auc = auc/csr.shape[0] #average auc score across all usres
    return auc



def precision_at_k(preds, csr, ks=(1,10,50)):
    """
    for each user, calculate precision @ K
    meaning: out of the top K recommended items for the user how many items were present in the ground truth set.
    Then average across all users

    :param preds: np.array (m- number of users, n- number of items) predictions for all users and items
    :param csr: scipy.sparse.csr_matrix() representation of validation set
    :return: dict.  keys: ks represents different k's. values:  float (0..1) precision score
    """

    pr_at_k = dict.fromkeys(ks,0)
    for user in range(csr.shape[0]):
        positives = csr[user].indices
        y_pred = preds[user]
        for k in ks:
            top_k = np.argsort(-y_pred)[:k] #sort item indices in desending order take top k
            act_set = set(positives)
            pred_set = set(top_k)
            pr_at_k[k] += len(act_set & pred_set) / len(pred_set)

    for k in ks:
        pr_at_k[k] /= csr.shape[0] #average across all users
    return pr_at_k


def hit_rate_at_k(preds, csr, ks=(1,10,50)):
    """
    for each user, calculate hit rate @ K
    meaning: is one of the top K recommended items for the user is present in the ground truth set? (0 or 1)
    Then average across all users

    :param preds: np.array (m- number of users, n- number of items) predictions for all users and items
    :param csr: scipy.sparse.csr_matrix() representation of validation set
    :return: dict.  keys: ks represents different k's. values:  float (0..1) precision score
    """

    hit_at_k = dict.fromkeys(ks,0)
    for user in range(csr.shape[0]):
        positive = csr[user].indices # we assume only one positive per user
        y_pred = preds[user]
        for k in ks:
            top_k = np.argsort(-y_pred)[:k] #sort item indices in desending order take top k
            if positive in top_k:
                hit_at_k[k] += 1.0

    for k in ks:
        hit_at_k[k] /= csr.shape[0] #average across all users
    return hit_at_k



def mpr(preds, csr, unranked_items):
    """
    mpr formula is given by  Sigma <u,i>  [r_ui * Rank_ui] / Sigma<u,i> [r_ui],
    Rank_ui is the percentile-ranking of item i within the ordered list of the catalog for user u.
    The catalog for user u is all of the items he did not rank (all non positive items for user u).
    r_ui is the ground truth ranking for item i by user u. Since all of our ground truth rankings are simply 1,
    We need to calculate Rank_ui for all <u,i> pairs and divide by the number of <u,i> pairs.



    :param preds: np.array (m- number of users, n- number of items) predictions for all users and items
    :param csr: scipy.sparse.csr_matrix() representation of validation set
    :param unranked_items dict <key: user_id, value: list of unranked item ids>
    :return: float (0..1) mpr score the lower the better
    """

    u_s, i_s = csr.nonzero()  # all <u,i> pairs in validation

    mpr = []
    for pair in zip(u_s, i_s):
        u,i = pair
        preds_for_user = preds[u,:]
        items_to_predict = [i]+unranked_items[u] # This is the catalog of the user. It includes item i and all of the unranked items for user u
        relevant_preds_for_user = preds_for_user[items_to_predict] #get the predictions for i and the catalog
        n = len(relevant_preds_for_user)
        rank = n - rankdata(relevant_preds_for_user)[0] # we want the rank of i, which we placed first in the array
        mpr.append(rank / n)


    res = sum(mpr)*1.0/len(mpr)
    return res



