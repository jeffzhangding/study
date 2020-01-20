__author__ = 'jeff'

"""简单讲架子搭出来了，"""


class JeffList(object):

    def __iter__(self):
        res = self.head
        while 1:
            if res is None:
                return
            else:
                yield res
            res = res.nxt

    def search(self, v):
        """"""
        for node in self:
            if node is None:
                return
            if node.value == v:
                return node

    def insert(self, k, v):
        """未完善"""
        node = self.head
        for i in range(k-2):
            node = node.nxt
        node.nxt = SingleNode(v)

    def append(self):
        """"""

    def get(self, k):
        """"""

    def delete(self, k):
        """"""

    def remove(self, v):
        """"""


class SingleNode(object):

    def __init__(self, value, nxt=None):
        """"""
        self.nxt = value or None
        self.value = None


class SingleList(JeffList):
    """"""
    def __init__(self):
        self.head = None


class DoubleNode(object):
    """"""

    def __init__(self, value=None, pre=None, nxt=None):
        self.value = value
        self.pre = pre
        self.nxt = nxt


class DoubleList(JeffList):
    """"""

    def __init__(self):
        self.head = None
        self.last = None





