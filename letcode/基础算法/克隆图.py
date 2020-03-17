__author__ = 'jeff'

"""
克隆图
"""

"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""


class Node:
    def __init__(self, val=0, neighbors=[]):
        self.val = val
        self.neighbors = neighbors


class Solution:

    def cloneGraph(self, node):
        """"""
        if node is None:
            return None
        book = {}
        res = self.dfs(node, book)
        return res

    def dfs(self, node, book):
        """"""
        c = Node(node.val, [])
        book[c.val] = c
        for n in node.neighbors:
            if n.val in book.keys():
                c.neighbors.append(book[n.val])
            else:
                c.neighbors.append(self.dfs(n, book))
        return c
