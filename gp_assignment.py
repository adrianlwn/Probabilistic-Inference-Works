import numpy as np
from scipy.optimize import minimize


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    # Task 1:
    # TODO: Implement a draw from a multivariate Gaussian here
    sample = np.random.multivariate_normal(mean,cov)
    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class LinearPlusRBF():
    def __init__(self, params):
        self.ln_sigma_b = params[0]
        self.ln_sigma_v = params[1]
        self.ln_sigma_f = params[2]
        self.ln_length_scale = params[3]
        self.ln_sigma_n = params[4]

        self.sigma2_b = np.exp(2*self.ln_sigma_b)
        self.sigma2_v = np.exp(2*self.ln_sigma_v)
        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_b = params[0]
        self.ln_sigma_v = params[1]
        self.ln_sigma_f = params[2]
        self.ln_length_scale = params[3]
        self.ln_sigma_n = params[4]

        self.sigma2_b = np.exp(2*self.ln_sigma_b)
        self.sigma2_v = np.exp(2*self.ln_sigma_v)
        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_b, self.ln_sigma_v, self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_b, self.sigma2_v, self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat1 = np.zeros((n,n))
        covMat2 = np.zeros((n,n))

        # Task 2:
        # TODO: Implement the covariance matrix here

        def k_linear(xp,xq):
            return self.sigma2_b + self.sigma2_v * np.dot(xp,xq)

        def k_RBF(xp,xq):
            return self.sigma2_f * np.exp( - np.linalg.norm(xp-xq,ord=2)**2 / float(2* self.length_scale**2) )

        covMat1 = np.array([[k_linear(X[i,:],X[j,:]) for i in range(n)] for j in range(n)])
        covMat2 = np.array([[k_RBF(X[i,:],X[j,:]) for i in range(n)] for j in range(n)])


        covMat = covMat1 + covMat2

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        return K
    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        Ka = self.k.covMatrix(self.X,Xa)
        Kxa = Ka[:self.X.shape[0],-Xa.shape[0]:]
        Kaa = Ka[-Xa.shape[0]:,-Xa.shape[0]:] - self.k.sigma2_n * np.eye(Xa.shape[0])

        L_inv = np.linalg.inv(self.L)
        K_inv = np.matmul(L_inv.transpose(),L_inv)

        mean_fa = np.matmul( np.transpose(Kxa) ,  np.matmul(K_inv ,self.y))
        cov_fa =  Kaa - np.matmul(np.matmul(np.transpose(Kxa),  K_inv  ), Kxa)
        # Return the mean and covariance

        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        L_inv = np.linalg.inv(self.L)
        K_inv = np.matmul(L_inv.transpose(),L_inv)
        s, log_det_K = np.linalg.slogdet(self.K)
        mll = float((np.matmul(np.matmul(np.transpose(self.y),K_inv),self.y) + log_det_K + self.n*np.log(2*np.pi))/2.)
        # Return mll
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters
        L_inv = np.linalg.inv(self.L)
        K_inv = np.matmul(L_inv.transpose(),L_inv)

        def gradcom(dK_sigma):
            g1 = - np.matmul(np.matmul(np.matmul(np.matmul(self.y.transpose(),K_inv),dK_sigma),K_inv),self.y)
            g2 = np.trace(np.matmul(K_inv,dK_sigma))
            return (g1 + g2)/2.0

        def dK_length_scale_f(xp,xq):
            norm2_pq = np.linalg.norm(xp - xq ,ord=2)**2
            return np.exp(2 * self.k.ln_sigma_f ) * norm2_pq * np.exp(-2*self.k.ln_length_scale) * np.exp( -(1/2.) * np.exp(-2*self.k.ln_length_scale)*norm2_pq )

        dK_sigma_b =  np.ones(self.K.shape)*2* np.exp(2*self.k.ln_sigma_b)
        dK_sigma_v =  np.ones(self.K.shape)*2* np.exp(2*self.k.ln_sigma_v) * np.matmul(self.X,self.X.transpose())
        dK_sigma_f = 2 * np.exp(2*self.k.ln_sigma_f) * np.array([[ np.exp( - np.linalg.norm(self.X[i,:]-self.X[j,:],ord=2)**2 / float(2* self.k.length_scale**2) ) for i in range(self.n)]for j in range(self.n)] )
        dK_length_scale =  np.array([[ dK_length_scale_f(self.X[i,:], self.X[j,:]) for i in range(self.n)]for j in range(self.n)] )
        dK_sigma_n = 2 * np.exp(2*self.k.ln_sigma_n) * np.eye(self.K.shape[0])

        grad_ln_sigma_b = float(gradcom(dK_sigma_b))
        grad_ln_sigma_v = float(gradcom(dK_sigma_v))
        grad_ln_sigma_f = float(gradcom(dK_sigma_f))
        grad_ln_length_scale = float(gradcom(dK_length_scale))
        grad_ln_sigma_n = float(gradcom(dK_sigma_n))


        # Combine gradients
        gradients = np.array([grad_ln_sigma_b, grad_ln_sigma_v, grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        mse = 1/float(len(ya)) * np.sum([(ya[i] - fbar[i])**2 for i in range(len(ya))])

        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        # Return msll
        pred_var = np.array([cov[i,i] + self.k.sigma2_n   for i in range(len(ya)) ] )
        def msll_i(i):
            return (1/2.)*np.log(2*np.pi*pred_var[i]) + (ya[i] - fbar[i])**2 / float(2*pred_var[i])

        msll = (1/float(ya.shape[0])) * np.sum([msll_i(i) for i in range(len(ya))])
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        print(res)
        return res.x

if __name__ == '__main__':

    np.random.seed(42)
    df = 'boston_housing.txt'
    X_train, y_train, X_test, y_test = loadData(df)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    params = [0,0,0, np.log(0.1), 0.5*np.log(0.5)]

    my_k = LinearPlusRBF(params=params)
    my_GP = GaussianProcessRegression(X=X_train, y=y_train, k=my_k)

    mean_fa, cov_fa = my_GP.predict(Xa=X_test)

    param_estim = my_GP.optimize(params=params,disp=True)

    my_GP.KMat(X=X_train, params=param_estim)

    mean_fa, cov_fa = my_GP.predict(Xa=X_test)
    mse = my_GP.mse(ya=y_test, fbar= mean_fa)
    msll = my_GP.msll(ya=y_test, fbar= mean_fa, cov=cov_fa)
    print('MSE = ' + str(mse))
    print('MSLL = ' + str(msll))








    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
