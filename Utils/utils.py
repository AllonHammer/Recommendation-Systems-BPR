import numpy as np
from sklearn.metrics import roc_auc_score

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
    auc = auc/csr.shape[0]
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

def mpr(preds, csr):
    """
    mpr formula is given by  Sigma <u,i>  [r_ui * Rank_ui] / Sigma<u,i> [r_ui],
    Rank_ui is the percentile-ranking of item i within the ordered list of all predictions for user u.
    r_ui is the ground truth ranking for item i by user u. Since all of our ground truth rankings are simply 1,
    We need to calculate Rank_ui for all <u,i> pairs and divide by the number of <u,i> pairs.


    :param preds: np.array (m- number of users, n- number of items) predictions for all users and items
    :param csr: scipy.sparse.csr_matrix() representation of validation set
    :return: float (0..1) mpr score the lower the better
    """

    u_s, i_s = csr.nonzero()  # all <u,i> pairs
    num_of_pairs = len(u_s) #could also be len(i_s)
    mpr = 0.0
    for pair in zip(u_s, i_s):
        user, item = pair[0], pair[1]
        y_pred=preds[user]
        ranks = np.argsort(-y_pred) / y_pred.shape[0] #sort item indices in decending order and rank them. 0 is the highert rank and 1 is the lowest
        mpr += ranks[item]
    mpr /= num_of_pairs
    return mpr
