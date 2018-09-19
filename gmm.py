import numpy as np
import matplotlib as plt
from scipy.stats import multivariate_normal


def phi(Y, mu_k, cov_k):
    '''
    the pdf of k ed component (gaussian normal distribution) in mixture model,
    row i represent proba of oberservation i appearing in k components
    '''
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

def e_step(Y, mu, cov, alpha):
    '''
    E step: calculate responsibilities
    :param Y: observatis matrix, every row represent a observationon
    :param mu: the mean of a component
    :param cov: cov matrix
    :param alpha: array of probabilitis of gaussian distribution component
    :return: matrix of responsibilities
    '''
    # the number of observations
    N = Y.shape[0]
    # the number of components
    K = alpha.shape[0]

    assert N>1, "there must be more than one observations"
    assert K>1, "there must be more than one components"

    # responsibilities matrix, row represent observation
    gamma = np.mat(np.zeros((N, K)))


    # calculate the probabilities of appearance of observations for  all components
    # row represents observation, column represent component
    prob = np.zeros((N, K))
    for k in range(K):
        # the k column of every row of prob array will be set phi value array.
        # phi value is the probability the observations belong to k component.
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # calculate the responsibilities of observations by components
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])

    return gamma


def m_step(Y, gamma):
    '''
    calculate parameters of mixture model
    :param Y: observatis matrix, every row represent a observationon
    :param gamma: reponsibilities matrix
    :return:
    '''
    # the number of observations, the number of features
    N, D = Y.shape
    # the number of components
    K = gamma.shape[1]

    # initialization of parameters
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # update parameters
    for k in range(K):
        # the sum of responsibilites for componets
        N_k = np.sum(gamma[:, k])
        # update mu
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d]))/ N_k
        # update covariance
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / N_k
        cov.append(cov_k)
        # update alpha
        alpha[k] = N_k / N

    cov = np.array(cov)
    return mu, cov, alpha


def scale_data(Y):
    '''
    scale data to 0 between 1
    :param Y: observations matrix
    :return: scaled observations matrix
    '''
    D = Y.shape[1]
    for i in range(D):
        max = Y[:, i].max()
        min = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min) / (max - min)
    return Y


def initialization_para(shape, K):
    '''
    initialization of parameters, the first step, and then iter
    :param shape: the shape of Y
    :param K: the number of components
    :return: initialized mu, cov and alpha
    '''
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha

def gmm_em(Y, K, times):
    '''
    implement gmm by em algorithm
    :param Y: observations matrix
    :param K: the number of components
    :param times: the iter times
    :return:
    '''
    Y = scale_data(Y)
    mu, cov, alpha = initialization_para(Y.shape, K)
    for i in range(times):
        gamma = e_step(Y, mu, cov, alpha)
        mu, cov, alpha = m_step(Y, gamma)
    return mu, cov, alpha



























