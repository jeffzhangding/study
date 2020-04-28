__author__ = 'jeff'

"""

第4章 朴素贝叶斯
1．朴素贝叶斯法是典型的生成学习方法。生成方法由训练数据学习联合概率分布 $P(X,Y)$，然后求得后验概率分布$P(Y|X)$。具体来说，利用训练数据学习$P(X|Y)$和$P(Y)$的估计，得到联合概率分布：

$$P(X,Y)＝P(Y)P(X|Y)$$
概率估计方法可以是极大似然估计或贝叶斯估计。

2．朴素贝叶斯法的基本假设是条件独立性，

$$\begin{aligned} P(X&amp;=x | Y=c_{k} )=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right) \\ &amp;=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right) \end{aligned}$$
这是一个较强的假设。由于这一假设，模型包含的条件概率的数量大为减少，朴素贝叶斯法的学习与预测大为简化。因而朴素贝叶斯法高效，且易于实现。其缺点是分类的性能不一定很高。

3．朴素贝叶斯法利用贝叶斯定理与学到的联合概率模型进行分类预测。

$$P(Y | X)=\frac{P(X, Y)}{P(X)}=\frac{P(Y) P(X | Y)}{\sum_{Y} P(Y) P(X | Y)}$$
将输入$x$分到后验概率最大的类$y$。

$$y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X_{j}=x^{(j)} | Y=c_{k}\right)$$
后验概率最大等价于0-1损失函数时的期望风险最小化。

模型：

高斯模型
多项式模型
伯努利模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from sklearn.naive_bayes import GaussianNB


def get_data():
    """"""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    return x_train, x_test, y_train, y_test

class NaiveBayes(object):
    """高斯朴素贝叶斯, $$P(x_i | y_k)=\frac{1}{\sqrt{2\pi\sigma^2_{yk}}}exp(-\frac{(x_i-\mu_{yk})^2}{2\sigma^2_{yk}})$$"""

    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.model = None
        self.y_probility = None

    @staticmethod
    def mean(x):
        """数学期望"""
        return np.mean(x, axis=0)

    @staticmethod
    def stdev(mean, x):
        """标准差（方差）"""
        # return math.sqrt(sum([pow(x - mean, 2) for x in X]) / float(len(X)))
        res = np.mean(np.square(x - mean), axis=0)
        return res

    # 概率密度函数
    @staticmethod
    def gaussian_probability(x, mu, sigma):
        """概率密度函数"""
        res = (1 / np.sqrt(2 * np.pi * np.square(sigma))) * np.exp(-np.square(x - mu) / (2 * np.square(sigma)))
        return res

    # def para(self, x):
    #     """"""
    #     mu = self.mean(x)
    #     sigma = self.stdev(mu, x)
    #     return mu, sigma

    def fit(self, x, y):
        """"""
        lenth = len(y)
        dt = {
           i: x[np.argwhere(y == i)[:, 0]] for i in set(y)
        }
        self.model = []

        for k, v in dt.items():
            mu = self.mean(v)
            sigma = self.stdev(mu, v)
            self.model.append((k, mu, sigma))

        self.y_probility = {
            k: len(v) / lenth for k, v in dt.items()
        }
        print('====gaussianNB train done!')
        return

    def predict(self, x):
        """"""
        # res = []
        p_arr, labels = self.caculate(x)
        index = np.argmax(p_arr, axis=1)
        res = [labels[i] for i in index]
        return res

    def caculate(self, x):
        """"""
        res = []
        labels = []
        for label, mu, sigma in self.model:
            p = self.gaussian_probability(x, mu, sigma)
            r = p[:, 0]
            for i in range(1, p.shape[1]):
                r *= p[:, i]
            labels.append(label)
            r *= self.y_probility[label]
            res.append(r)
        return np.stack(res, axis=1), labels

    def score(self, x, y):
        """"""
        predict_y = self.predict(x)
        s = np.where(np.array(predict_y) == np.array(y), 1, 0)
        return np.sum(s) / len(s)


def jeff_bayes():
    """"""
    x_train, x_test, y_train, y_test = get_data()
    model = NaiveBayes()
    model.fit(x_train, y_train)
    s = model.score(x_test, y_test)
    print('===score: %s' % s)


def skit_bayes():
    """"""
    x_train, x_test, y_train, y_test = get_data()
    model = GaussianNB()
    model.fit(x_train, y_train)
    s = model.score(x_test, y_test)
    print('===score: %s' % s)


if __name__ == '__main__':
    # jeff_bayes()
    skit_bayes()
