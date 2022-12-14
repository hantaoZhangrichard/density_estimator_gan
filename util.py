import numpy as np


class GMM_sampler():
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None
            self.mean = np.random.uniform(-5, 5, (self.n_components, self.dim))
        else:
            assert cov is not None
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=self.weights)
        if mean is None:
            self.X = np.array([np.random.normal(self.mean[i], scale=self.sd) for i in self.Y], dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i], cov=self.cov[i]) for i in self.Y], dtype='float64')
        self.X_train, self.X_val, self.X_test = self.split(self.X)

    def split(self, data):
        N_test = 2000
        data_test = data[-N_test:]
        data = data[0:-N_test]

        N_validate = 2000
        data_validate = data[-N_validate:0]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def train(self, batch_size, label=False):
        indx = np.random.randint(low=0, high=len(self.X_train), size=batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y


class Gaussian_sampler():
    def __init__(self, mean, sd=1, N=10000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size, len(self.mean)))
        self.Y = None

    def train(self, batch_size):
        indx = np.random.randint(low=0, high=self.total_size, size=batch_size)
        return self.X[indx, :]

    def get_batch(self, batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size, len(self.mean)))

    def load_all(self):
        return self.X, self.Y


# each dim is a gmm

class GMM_indep_sampler():
    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.bound = bound
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val, self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y = None

    def generate_gmm(self, weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i], self.sd) for i in Y], dtype='float64')

    def split(self, data):
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test

    def get_density(self, data):
        assert data.shape[1] == self.dim
        from scipy.stats import norm
        centers = np.linspace(-self.bound, self.bound, self.n_components)
        prob = []
        for i in range(self.dim):
            p_mat = np.zeros((self.n_components, len(data)))
            for j in range(len(data)):
                for k in range(self.n_components):
                    p_mat[k, j] = norm.pdf(data[j, i], loc=centers[k], scale=self.sd)
            prob.append(np.mean(p_mat, axis=0))
        prob = np.stack(prob)
        return np.prod(prob, axis=0)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y
