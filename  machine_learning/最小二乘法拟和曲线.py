__author__ = 'jeff'

"""

1．统计学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行分析与预测的一门学科。统计学习包括监督学习、非监督学习、半监督学习和强化学习。

2．统计学习方法三要素——模型、策略、算法，对理解统计学习方法起到提纲挈领的作用。

3．本书主要讨论监督学习，监督学习可以概括如下：从给定有限的训练数据出发， 假设数据是独立同分布的，而且假设模型属于某个假设空间，应用某一评价准则，从假设空间中选取一个最优的模型，使它对已给训练数据及未知测试数据在给定评价标准意义下有最准确的预测。

4．统计学习中，进行模型选择或者说提高学习的泛化能力是一个重要问题。如果只考虑减少训练误差，就可能产生过拟合现象。模型选择的方法有正则化与交叉验证。学习方法泛化能力的分析是统计学习理论研究的重要课题。

5．分类问题、标注问题和回归问题都是监督学习的重要问题。本书中介绍的统计学习方法包括感知机、$k$近邻法、朴素贝叶斯法、决策树、逻辑斯谛回归与最大熵模型、支持向量机、提升方法、EM算法、隐马尔可夫模型和条件随机场。这些方法是主要的分类、标注以及回归方法。它们又可以归类为生成方法与判别方法。

使用最小二乘法拟和曲线
高斯于1823年在误差$e_1,…,e_n$独立同分布的假定下,证明了最小二乘方法的一个最优性质: 在所有无偏的线性估计类中,最小二乘方法是其中方差最小的！ 对于数据$(x_i, y_i)   (i=1, 2, 3...,m)$

拟合出函数$h(x)$

有误差，即残差：$r_i=h(x_i)-y_i$

此时$L2$范数(残差平方和)最小时，$h(x)$ 和 $y$ 相似度最高，更拟合

一般的$H(x)$为$n$次的多项式，$H(x)=w_0+w_1x+w_2x^2+...w_nx^n$

$w(w_0,w_1,w_2,...,w_n)$为参数

最小二乘法就是要找到一组 $w(w_0,w_1,w_2,...,w_n)$ ，使得$\sum_{i=1}^n(h(x_i)-y_i)^2$ (残差平方和) 最小

即，求 $min\sum_{i=1}^n(h(x_i)-y_i)^2$

"""


import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def real_func(x):
    """原函数"""
    return np.sin(2*np.pi*x)


def fit_func(p, x):
    """"""
    f = np.poly1d(p)
    return f(x)


def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 100)

y_ = real_func(x)
y = [np.random.normal(0, 0.1) + yl for yl in y_]


def fitting(m=0):
    """"""
    p_init = np.random.rand(m+1)

    # 最小二乘法的参数计算
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='model')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq


regularization = 0.001


def residuals_func_regularization(p, x, y):
    """"""
    ret = fit_func(p, x) - y
    ret = np.append(ret,
                    np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
    return ret


def resid_fit(m):
    """"""
    p_init = np.random.rand(m + 1)
    p_lsq_regularization = leastsq(
        residuals_func_regularization, p_init, args=(x, y))

    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
    plt.plot(
        x_points,
        fit_func(p_lsq_regularization[0], x_points),
        label='regularization')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()


if __name__ == '__main__':
    # fitting(0)
    fitting(9)
    # plt.show()


