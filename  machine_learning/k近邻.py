__author__ = 'jeff'

"""
1．k近邻法是基本且简单的分类与回归方法。k近邻法的基本做法是：对给定的训练实例点和输入实例点，首先确定输入实例点的k个最近邻训练实例点，然后利用这k个训练实例点的类的多数来预测输入实例点的类。

2．k近邻模型对应于基于训练数据集对特征空间的一个划分。k近邻法中，当训练集、距离度量、k值及分类决策规则确定后，其结果唯一确定。

3．k近邻法三要素：距离度量、k值的选择和分类决策规则。常用的距离度量是欧氏距离及更一般的pL距离。k值小时，k近邻模型更复杂；k值大时，k近邻模型更简单。k值的选择反映了对近似误差与估计误差之间的权衡，通常由交叉验证选择最优的k。

常用的分类决策规则是多数表决，对应于经验风险最小化。

4．k近邻法的实现需要考虑如何快速搜索k个最近邻点。kd树是一种便于对k维空间中的数据进行快速检索的数据结构。kd树是二叉树，表示对k维空间的一个划分，其每个结点对应于k维空间划分中的一个超矩形区域。利用kd树可以省去对大部分数据点的搜索， 从而减少搜索的计算量。

距离度量
设特征空间x是n维实数向量空间 ，x_{i}, x_{j} \in \mathcal{X},x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(n)}\right)^{\mathrm{T}},x_{j}=\left(x_{j}^{(1)}, x_{j}^{(2)}, \cdots, x_{j}^{(n)}\right)^{\mathrm{T}} ，则：x_i,x_j的L_p距离定义为:

L_{p}\left(x_{i}, x_{j}\right)=\left(\sum_{i=1}^{n}\left|x_{i}^{(i)}-x_{j}^{(l)}\right|^{p}\right)^{\frac{1}{p}}

p= 1 曼哈顿距离
p= 2 欧氏距离
p= \infty 切比雪夫距离

"""


from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class Data(object):
    """"""

    def __init__(self):
        """"""
        self.df = self.iris()

    def get_test_data(self):
        """"""

    def iris(self):
        """"""
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
        # self.df = df
        return df

    def show(self):
        plt.scatter(self.df[:50]['sepal length'], self.df[:50]['sepal width'], label='0')
        plt.scatter(self.df[50:100]['sepal length'], self.df[50:100]['sepal width'], label='1')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.legend()
        plt.show()


class Knn(object):
    """"""

    def fit(self):
        """"""


class SkitKnn(object):
    """"""


if __name__ == '__main__':
    d = Data()

    print(d.df)
    d.show()

