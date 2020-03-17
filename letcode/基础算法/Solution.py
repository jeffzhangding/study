"""
岛屿问题
"""

from queue import Queue


class MyCircularQueue(object):

    def __init__(self, k=100):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.q = [None for i in range(k + 1)]
        self.head = 0
        self.tail = 0
        self.k = k

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if self.isFull():
            return False
        else:
            self.tail = self.next_index(self.tail)
            self.q[self.tail] = value
            return True

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.isEmpty():
            return False
        else:
            self.head = self.next_index(self.head)
            return True

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            next_head = self.next_index(self.head)
            return self.q[next_head]

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            res = self.q[self.tail]
            # self.tail = self.last_index(self.tail)
            return res

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return self.head == self.tail

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return self.next_index(self.tail) == self.head

    def next_index(self, index):
        """
        """
        if index == self.k:
            return 0
        else:
            return index + 1

    def last_index(self, index):
        if index == 0:
            return self.k
        else:
            return index - 1

    def empty(self):
        return self.isEmpty()

    def get(self):
        res = self.Front()
        self.deQueue()
        return res

    def put(self, value):
        return self.enQueue(value)


class Solution(object):

    def __init__(self):
        super(Solution, self).__init__()
        # self.q = Queue()
        self.q = MyCircularQueue()

    def _get_node_list(self, grid):
        """"""
        node_list = []
        i = 0
        for row in grid:
            j = 0
            for node in row:
                if node == '1':
                    node_list.append((i, j))
                j += 1
            i += 1
        return node_list

    def _get_arounds(self, node, grid):
        """获取周围的点"""
        try:
            k = grid[node[0]][node[1]]
        except Exception as e:
            k = '0'
        return k

    def _bfs(self, node, node_set, grid):
        """"""
        self.q.put(node)
        while not self.q.empty():
            node = self.q.get()
            round_list = [(node[0]+1, node[1]), (node[0]-1, node[1]), (node[0], node[1]+1), (node[0], node[1]-1)]
            for r_node in round_list:
                if r_node in node_set:
                    self.q.put(r_node)
                    node_set.remove(r_node)

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        node_set = set(self._get_node_list(grid))
        res = 0
        while len(node_set) > 0:
            res += 1
            node = node_set.pop()
            self._bfs(node, node_set, grid)
        return res


if __name__ == '__main__':

    x = """11110
11010
11000
00000"""
#     x = """11000
# 11000
# 00100
# 00011"""
    g = []
    for i in x.split('\n'):
        g.append(list(i))
    res = Solution().numIslands(g)
    print(res)

