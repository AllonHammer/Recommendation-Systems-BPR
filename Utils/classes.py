import pandas as pd
from os.path import dirname, abspath, join
import numpy as np
from tqdm import trange
from scipy.sparse import csr_matrix
from hyperopt import hp, fmin, tpe,  Trials
from Utils.utils import item_popularity, sample_batch_uij, auc_score

class DataSet():
    """ A class used to represent a dataset"""
    def __init__(self,  seed=12, validation_percent=0.3):
        #
        parent_path= dirname(dirname(abspath(__file__)))

        self.data = pd.read_csv(join(parent_path,'Resources/Train.csv'))
        self.seed = seed
        self.popularity_test = pd.read_csv(join(parent_path,'Resources/PopularityTest.csv'))
        self.random_test = pd.read_csv(join(parent_path,'Resources/RandomTest.csv'))
        self.validation_percent = validation_percent
        self.train, self.validation=self.train_validation()
        self.item_popularity_train=item_popularity(self.train)
        self.item_popularity_validation=item_popularity(self.validation)
        self.users = self.train.shape[0]
        self.items = self.train.shape[1]



    def train_validation(self):
        seed=np.random.RandomState(self.seed)
        #create SCR matrix for efficiency
        rows = self.data.UserID.astype('category').cat.codes
        cols = self.data.ItemID.astype('category').cat.codes
        csr = csr_matrix((np.ones(self.data.shape[0]), (rows, cols)))
        train = csr.copy()
        validation  = csr_matrix(csr.shape).tolil() # all zeros lil_matrix
        #split to train and test by user_id
        users = csr.shape[0]
        for user in range(users):
            # csr.indices is an array of all of the non zero elements
            # csr.indptr  is an array. its indexes represent users and the values are pointers to the
            # indexes of csr.indeces where the data of the relevant user is
            # https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr #
            items = csr.indices[csr.indptr[user]: csr.indptr[user+1]]
            #for each user randomaly select items for validation set
            cutoff = int(self.validation_percent*len(items))
            items_chosen = seed.choice(items, size=cutoff, replace=False)
            train[user,items_chosen] = 0
            validation[user,items_chosen] = 1

        return train, validation.tocsr()






class Model():
    """ A class used to represent a model

    :arg latent_dim: int the size of the latent dimension d
    :arg n_users: int the number of users in the training set
    :arg n_items: mapping  the number of items in the training set
    :arg l_users: float regularization term for users
    :arg l_items: float regularization term for items
    :arg learning_rate: float
    :arg seed: int random seed for reproduction
    """

    def __init__(self, latent_dim, n_users, n_items, l_users, l_items, l_bias_items, learning_rate=0.01, use_decay=False,
                 learning_decay=0.001, seed=12):
        self.k = latent_dim
        self.m = n_users
        self.n = n_items
        self.lu = l_users
        self.lv = l_items
        self.lvb = l_bias_items
        self.alpha = learning_rate
        self.use_decay = use_decay
        self.decay = learning_decay
        self.seed = seed

    def init_variables(self):
        """init latent user and item matrices U,V"""

        rstate = np.random.RandomState(self.seed)
        k=int(self.k)
        self.u = rstate.normal(0, 1 / np.sqrt(k), (self.m, k))  #switch variance to 0.1?
        self.v = rstate.normal(0, 1 / np.sqrt(k), (self.n, k))  #switch variance to 0.1?
        self.b = rstate.normal(0,1, size=self.n)


    def fit(self, train, validation=None, epochs=100, batch_size=1,  sample_method='random', verbose=True):
        """

        :param train: scipy.sparse.csr_matrix() of train set
        :param validation: scipy.sparse.csr_matrix() of validation set
        :param epochs: int, one epoch is a full update across all users
        :param batch_size: int
        :param sample_method: string 'random'/'populairty
        :return best auc: float (0..1) the higher the better
        """

        self.init_variables()

        pop_proba = item_popularity(train)
        batch_iters = self.m // batch_size

        if verbose:
            epoch_loop = trange(epochs, desc = self.__class__.__name__)
        else:
            epoch_loop = range(epochs)
        for e in epoch_loop:
            for _ in range(batch_iters):
                batch_u, batch_i, batch_j = sample_batch_uij(batch_size=batch_size, n_users=self.m,
                                                             n_items=self.n,
                                                             indices=train.indices, indptr=train.indptr,
                                                             proba_lst=pop_proba,
                                                             sample_by=sample_method)
                self.update_step(batch_u, batch_i, batch_j)
            if self.use_decay:
                self.alpha = self.alpha * (1 / (1 + self.decay * e))

        if validation is None:
            return self

        # when called from self.hyperparmas_tune()
        preds = self.predict_all(use_bias=True)
        auc=auc_score(preds, validation)


        return np.round(auc,3)



    def update_step(self, batch_u, batch_i, batch_j):
        """ computes gradients of loss function and updates latent user and items matrices U,V

        :param batch_u: np.array (batch_size,) of users
        :param batch_i: np.array (batch_size,) of positive items
        :param batch_j: np.array (batch_size,) of negative items
        """
        u = self.u[batch_u, :] # (batch_size, k)
        i = self.v[batch_i, :] # (batch_size, k)
        j = self.v[batch_j, :] # (batch_size, k)
        bi = self.b[batch_i]
        bj = self.b[batch_j]

        # given a single triplet <u1,i1,j1> we are only interested in the interaction between <u1,i1> and  <u1,j1>
        # and NOT the interaction between  <u1,i2> or <u1,j2> for example. Therefore by doing   np.dot(u,v.T) we get
        # many indices that we are not interested in. We are only interested in the DIAGONAL of np.dot(u,v.T). This is
        # exactly what batch_product means. So [u11*v11+u12*v12+...+u1k*v1k
        #                                       u21*v21+u22*v22+...+u2k*v2k
        #                                       ..........................
        #                                       ub1*vb1+ub2*vb2+...+ubk*vbk]
        # this has a dimension of (batch_size,)
        # it is equivalent to do element wise product between u,v and summing on columns
        #                                      [u11*v11,         u12*v12,         .... u1k*v1k]
        #                                      [u21*v21,         u22*v22,         .... u2k*v2k]
        #                                      ................................................
        #                                      [ub1*vb1,         ub2*vb2,         .... ubk*vbk]
        # which has a dimension of (batch_size, k ), now sum on columns
        # we get exactly                       [u11*v11+u12*v12+...+u1k*v1k
        #                                       u21*v21+u22*v22+...+u2k*v2k
        #                                       ..........................
        #                                       ub1*vb1+ub2*vb2+...+ubk*vbk]
        # which is like above with dimension of (batch_size,)
        # so we need x_ui = np.sum(u*i, axis=1)
        # and        x_uj = np.sum(u*j, axis=1)
        # to do      x_uij = xui-xuj
        # so         x_uij = np.sum(u*(i-j), axis=1)

        x_uij = np.sum(u*(i-j), axis=1) # (batch_size,)

        # compute 1-sigmoid(x_uij)
        err = np.exp(-x_uij)/(1+np.exp(-x_uij)) # (batch_size,)
        err = np.reshape(err,(-1,1)) # reshape for broadcasting ,shape is (batch_size, 1)
        # gradients of err w.r.t  u,i,j
        grad_err_u = (i-j)       #shape is (batch_size, k)
        grad_err_i = u           #shape is (batch_size, k)
        grad_err_j = -u          #shape is (batch_size, k)
        grad_err_bi = 1
        grad_err_bj = -1

        #gradients of log posterior P(D/theta)*P(theta) w.r.t  u,i,j
        grad_u =  err * grad_err_u - self.lu * u  # (batch_size, 1) --> broadcast  *  (batch_size, k) - scalar * (batch_size, k)
        grad_i =  err * grad_err_i - self.lv * i  # (batch_size, 1) --> broadcast  *  (batch_size, k) - scalar * (batch_size, k)
        grad_j =  err * grad_err_j - self.lv * j  # (batch_size, 1) --> broadcast  *  (batch_size, k) - scalar * (batch_size, k)
        grad_bi = (err * grad_err_bi).reshape(-1) - (self.lvb * bi)  # (batch_size, ) -scalar * (batch_size,)
        grad_bj = (err * grad_err_bj).reshape(-1) - (self.lvb * bj)  # (batch_size, ) -scalar * (batch_size,)

        # we want to maximize the log posterior therefore we are going WITH the gradient
        self.u[batch_u] += self.alpha * grad_u
        self.v[batch_i] += self.alpha * grad_i
        self.v[batch_j] += self.alpha * grad_j
        self.b[batch_i] += self.alpha * grad_bi
        self.b[batch_j] += self.alpha * grad_bj


    def predict(self,u, i, j, use_bias=False):
        """ predict that for user u, the probability it will prefer item i over j
        :param u: int. user_id
        :param i: int. positive item_id
        :param j: int. negative item_id
        """
        if i>self.v.shape[0] or j>self.v.shape[0]: # unknown item
            print('Unknown items {}, {}'.format(i,j))
            return 0.5
        if use_bias:
            x = np.dot(self.u[u, :], self.v[i, :].T) + self.b[i] - np.dot(self.u[u, :], self.v[j, :].T) - self.b[j]
        else:
            x = np.dot(self.u[u, :], self.v[i, :].T) - np.dot(self.u[u, :], self.v[j, :].T)

        return 1/(1+np.exp(-x))

    def predict_all(self, use_bias=False):
        """ predict all items for all users"""
        if use_bias:
            b = self.b
        else:
            b = 0
        x= np.dot(self.u, self.v.T)+b #(m- number of users, n- number of items)
        return 1 / (1 + np.exp(-x))











    def hyperparmas_tune(self, train, validation,  epochs=100, batch_size=100,  sample_method='random', max_evals=100):
        """Tunes hyperparams in bayesian optimization:

        :param train: scipy.sparse.csr_matrix() of train set
        :param validation: scipy.sparse.csr_matrix() of validation set
        :param epochs: int, one epoch is a full update across all users
        :param batch_size: int
        :param sample_method: string 'random'/'populairty
        :param max_evals: int number of iterations of bayesian optimization
        :return dict <key: hyperparam, value: best value>

        """
        payload={}
        params = {
            'use_decay': hp.choice('use_decay', [False,True]),
            'k': hp.quniform('k', 5,50,1),
            'lu': hp.loguniform('lu', -4.5, -1),
            'lv': hp.loguniform('li', -4.5, -1),
            'lvb': hp.loguniform('lvb', -4.5, -1),
            'alpha': hp.loguniform('alpha', -4.5, -2),
            'decay': hp.loguniform('decay', -5.5,-4.5)
        }
        payload['params']=params
        payload['external']={'train': train, 'validation': validation,  'epochs': epochs, 'batch_size': batch_size,  'sample_method': sample_method, 'verbose': False}
        trials = Trials()
        best = fmin(fn=self.objective, space=payload, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best


    def objective(self, payload):
        """ helper function for hyperparams_tune
            :param payload: dict <key: params ,  value: dict of hyperparams
                                  key: external, valueL dict of external data>
            :return float"""

        params=payload['params']
        external=payload['external']
        for key in params.keys():
            exec("self.{}={}".format(key, params[key]))
        print(' Set Hyperparams to k:{} lu: {} lv:{} lvb:{} alpha:{} decay: {} use_decay: {}'.format(self.k,self.lu,self.lv, self.lvb, self.alpha, self.decay, self.use_decay))
        auc = self.fit(**external)
        auc *= -1 #minimizing the negative
        print('AUC  Validation set: {}'.format(-1*auc))
        return auc










