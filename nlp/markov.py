__author__ = 'jeff'

import numpy as np
import random
# from math import sqrt
import matplotlib.pylab as plt
import math
from scipy.stats import norm


def markov():
    """马尔科夫链稳态"""
    init_array = np.array([0.21, 0.68, 0.11])
    transfer_matrix = np.array([[0.65, 0.28, 0.07],
                               [0.15, 0.67, 0.18],
                               [0.12, 0.36, 0.52]])
    restmp = init_array
    for i in range(25):
        res = np.dot(restmp, transfer_matrix)
        print(i, "\t", res)
        restmp = res


def caculate_pi(n):
    """通过蒙特卡洛算法计算圆周率"""
    inner_number = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        l = x*x + y*y
        if l < 1:
            inner_number += 1

    pi = 4 * inner_number / n
    print('圆周率估计===%s' % str(pi))


def mcmc():
    mu = 3
    sigma = 10

    # 转移矩阵Q,因为是模拟数字，只有一维，所以Q是个数字(1*1)
    def q(x):
        return np.exp(-(x - mu) ** 2 / (sigma ** 2))

    # 按照转移矩阵Q生成样本
    def qsample():
        return np.random.normal(mu, sigma)

    # 目标分布函数p(x)
    def p(x):
        return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)

    def mcmcsample(n=20000):
        sample = np.zeros(n)
        sample[0] = 0.5  # 初始化
        for i in range(n - 1):
            qs = qsample()  # 从转移矩阵Q(x)得到样本xt
            # u = np.random.rand()  # 均匀分布
            # u = random.uniform(1, 2)
            u = random.random()
            # alpha_i_j = (p(qs) * q(sample[i])) / (p(sample[i]) * qs)  # alpha(i, j)表达式
            alpha_i_j = p(qs) / p(sample[i])  # alpha(i, j)表达式
            if alpha_i_j > 1:
                print('====')
                # continue
            if u < min(alpha_i_j, 1):
                sample[i + 1] = qs  # 接受
            else:
                sample[i + 1] = sample[i]  # 拒绝

        return sample

    x = np.arange(0, 4, 0.1)
    realdata = p(x)
    sampledata = mcmcsample()
    plt.plot(x, realdata, 'g', lw=3)  # 理想数据
    plt.plot(x, q(x), 'r')  # Q(x)转移矩阵的数据
    plt.hist(sampledata, bins=x, normed=1, fc='c')  # 采样生成的数据
    plt.show()


def mcmc2():
    def norm_dist_prob(theta):
        y = norm.pdf(theta, loc=3, scale=2)
        return y

    T = 5000
    pi = [0 for i in range(T)]
    sigma = 1
    t = 0
    while t < T - 1:
        t = t + 1
        pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
        alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))

        u = random.uniform(0, 1)
        if u < alpha:
            pi[t] = pi_star[0]
        else:
            pi[t] = pi[t - 1]

    plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
    num_bins = 50
    plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    # markov()
    # caculate_pi(1000000)
    mcmc()
    # mcmc2()