import numpy as np
import numpy.random as rn
from scipy import optimize, stats
import scipy.linalg as linalg


# ##############################################################################
# load_data generates a binary dataset for visualisation and testing using two
# parameters:
# * A **jitter** parameter that controls how noisy the data are; and
# * An **offset** parameter that controls the separation between the two classes.
#
# Do not change this function!
# ##############################################################################
def load_data(N=50, jitter=0.7, offset=1.2):
    # Generate the data
    x = np.vstack([rn.normal(0, jitter, (N // 2, 1)),
                   rn.normal(offset, jitter, (N // 2, 1))])
    y = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    x_test = np.linspace(-2, offset + 2).reshape(-1, 1)

    # Make the augmented data matrix by adding a column of ones
    x_train = np.hstack([np.ones((N, 1)), x])
    x_test = np.hstack([np.ones((N, 1)), x_test])
    return x_train, y, x_test


def sigmoid(x):
    return 1/(1+np.exp(-x))
# ##############################################################################
# predict takes a input matrix X and parameters of the logistic regression theta
# and predicts the output of the logistic regression.
# ##############################################################################
def predict(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: prediction of f(X); K x 1 vector

    # Task 1:
    # TODO: Implement the prediction of a logistic regression here.

    prediction = sigmoid(np.matmul(X,theta))

    return prediction


def predict_binary(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: binary prediction of f(X); K x 1 vector; should be 0 or 1

    prediction = 1. * (predict(X, theta) > 0.5)

    return prediction


# ##############################################################################
# log_likelihood takes data matrices x and y and parameters of the logistic
# regression theta and returns the log likelihood of the data given the logistic
# regression.
# ##############################################################################
def log_likelihood(X, y, theta):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # theta: parameters (D x 1)
    # returns: log likelihood, scalar

    # Task 2:
    # TODO: Calculate the log-likelihood of a dataset
    # given a value of theta.
    def L_i(X_i,y_i):
        sig = predict(X_i.transpose(),theta)
        return y_i*np.log(sig) + (1-y_i)*np.log(1-sig)

    L = np.sum([L_i(X[i,:],y[i]) for i in range(X.shape[0])])

    return L


# ##############################################################################
# max_lik_estimate takes data matrices x and y ands return the maximum
# likelihood parameters of a logistic regression.
# ##############################################################################
def max_lik_estimate(X, y):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_ml = theta_init

    # Task 3:
    # TODO: Optimize the log-likelihood function you've
    # written above an obtain a maximum likelihood estimate
    def neg_log_ll(theta):
        return -log_likelihood(X,y,theta)

    optML = optimize.minimize(  fun=neg_log_ll,
                        x0=theta_init ,
                        method='BFGS', )
    theta_ml = optML.x
    return theta_ml


# ##############################################################################
# neg_log_posterior takes data matrices x and y and parameters of the logistic
# regression theta as well as a prior mean m and covariance S and returns the
# negative log posterior of the data given the logistic regression.
# ##############################################################################
def neg_log_posterior(theta, X, y, m, S):
    # theta: D x 1 matrix of parameters
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: scalar negative log posterior
    negative_log_posterior = 0

    # Task 4:
    # TODO: Calculate the log-posterior

    prior_theta = stats.multivariate_normal(mean=m.flatten(-1), cov=S)
    #print(prior_theta,theta.shape,prior_theta.logpdf(theta),log_likelihood(X, y, theta))
    negative_log_posterior = -(log_likelihood(X, y, theta) + prior_theta.logpdf(theta.flatten(-1) ) + np.log(2*np.pi) + np.log(linalg.det(S)) )
    return negative_log_posterior


# ##############################################################################
# map_estimate takes data matrices x and y as well as a prior mean m and
# covariance  and returns the maximum a posteriori parameters of a logistic
# regression.
# ##############################################################################
def map_estimate(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_map = theta_init

    # Task 5:
    # TODO: Optimize the log-posterior function you've
    # written above an obtain a maximum a posteriori estimate
    def neg_log_map(theta):
        return neg_log_posterior(theta, X, y, m, S)

    optMAP = optimize.minimize(  fun=neg_log_map,
                        x0=theta_init ,
                        method='BFGS', )
    theta_map = optMAP.x
    return theta_map


# ##############################################################################
# laplace_q takes an array of points z and returns an array with Laplace
# approximation q evaluated at all points in z.
# ##############################################################################
def laplace_q(z):
    # z: double array of size (T,)
    # returns: array with Laplace approximation q evaluated
    #          at all points in z
    q = np.zeros_like(z)

    # Task 6:
    # TODO: Evaluate the Laplace approximation $q(z)$.
    z_star = 2
    A_1 = 4
    q_dist = stats.multivariate_normal(mean=z_star,cov=A_1)

    q = q_dist.pdf(z.flatten(-1))
    return q


# ##############################################################################
# get_posterior takes data matrices x and y as well as a prior mean m and
# covariance and returns the maximum a posteriori solution to parameters
# of a logistic regression as well as the covariance approximated with the
# Laplace approximation.
# ##############################################################################
def get_posterior(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)
    #          covariance of Laplace approximation (D x D)

    mu_post = np.zeros_like(m)
    S_post = np.zeros_like(S)

    # Task 7:
    # TODO: Calculate the Laplace approximation of p(theta | X, y)

    mu_post = map_estimate(X, y, m, S)
    def S_i(X_i,y_i):
        return y_i*(1-y_i)*np.matmul(X_i.transpose(),X_i)
    S_ = np.sum(np.array([S_i(X[i,:],y[i]) for i in range(X.shape[0])]),axis=0)
    S_post = linalg.inv(S) + S_#np.einsum('qu,qk,qp -> kp',y*(1-y),X,X)

    return mu_post, S_post


# ##############################################################################
# metropolis_hastings_sample takes data matrices x and y as well as a prior mean
# m and covariance and the number of iterations of a sampling process.
# It returns the sampling chain of the parameters of the logistic regression
# using the Metropolis algorithm.
# ##############################################################################
def metropolis_hastings_sample(X, y, m, S, nb_iter):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: nb_iter x D matrix of posterior samples

    D = X.shape[1]
    samples = np.zeros((nb_iter, D))

    # Task 8:
    # TODO: Write a function to sample from the posterior of the
    # parameters of the logistic regression p(theta | X, y) using the
    # Metropolis algorithm.


    return samples
