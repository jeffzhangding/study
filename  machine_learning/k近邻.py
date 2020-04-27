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
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


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

    @property
    def data(self):
        iris = load_iris()
        # data = np.array(self.df.iloc[:100, [0, 1, -1]])
        data = iris['data']
        target = iris['target']
        x, y = data[::2], target[::2]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        return x_train, x_test, y_train, y_test


class Knn(object):
    """"""

    def __init__(self, x_train, y_train, k=3, p=2):
        """"""
        self.k = k
        self.p = p
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x):
        """"""
        dist = np.linalg.norm(x - self.x_train, axis=1, ord=self.p)
        index = np.argpartition(dist, self.k)[:self.k]
        y = self.y_train[index]
        r_dict = Counter(y)
        res = max(r_dict.items(), key=lambda v: v[1])
        return res[0]

    def score(self, x_test, y_test):
        """"""
        sucess = 0.0
        for i in range(len(x_test)):
            res = self.predict(x_test[i])
            print('pridict %s ===  lable: %s ' % (res, y_test[i]))
            if res == y_test[i]:
                sucess += 1
        return sucess / len(x_test)

    def skt_data(self):
        """"""
        d = Data()
        x_train, x_test, y_train, y_test = d.data
        print(d.df)
        d.show()
        model = Knn(x_train, y_train, k=10, p=2)
        score = model.score(x_test, y_test)
        print(score)


def knn_test():
    d = Data()
    x_train, x_test, y_train, y_test = d.data
    # print(d.df)
    # d.show()
    model = Knn(x_train, y_train, k=10, p=2)
    score = model.score(x_test, y_test)
    print('jeff knn mode: == %s' % str(score))


class SkitKnn(object):
    """"""

    def __init__(self, x_train, y_train, k=None, weights='distance'):
        """"""
        self.model = neighbors.KNeighborsClassifier(k, weights=weights)
        self.model.fit(x_train, y_train)

    def predict(self, x, show=True):
        """"""
        # x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        # y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                      np.arange(y_min, y_max, h))
        res = self.model.predict([x])
        return res

    def score(self, x_test, y_test):
        """"""
        sucess = 0.0
        for i in range(len(x_test)):
            res = self.predict(x_test[i])
            print('pridict %s ===  lable: %s ' % (res, y_test[i]))
            if res == y_test[i]:
                sucess += 1
        return sucess / len(x_test)


def skt_test():
    """"""
    d = Data()
    x_train, x_test, y_train, y_test = d.data
    # print(d.df)
    # d.show()
    model = SkitKnn(x_train, y_train, k=10)
    score = model.score(x_test, y_test)
    # score = model.model.score(x_test, y_test)
    print('skit mode: == %s' % str(score))


if __name__ == '__main__':
    # skt_test()
    knn_test()


