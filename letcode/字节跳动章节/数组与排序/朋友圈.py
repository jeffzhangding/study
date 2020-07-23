__author__ = 'jeff'

"""
班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。

给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。

示例 1:

输入: 
[[1,1,0],
 [1,1,0],
 [0,0,1]]
输出: 2 
说明：已知学生0和学生1互为朋友，他们在一个朋友圈。
第2个学生自己在一个朋友圈。所以返回2。
示例 2:

输入: 
[[1,1,0],
 [1,1,1],
 [0,1,1]]
输出: 1
说明：已知学生0和学生1互为朋友，学生1和学生2互为朋友，所以学生0和学生2也是朋友，所以他们三个在一个朋友圈，返回1。
注意：

N 在[1,200]的范围内。
对于所有学生，有M[i][i] = 1。
如果有M[i][j] = 1，则有M[j][i] = 1。

"""

"""
解题思路：
上面的问题其实是一个遍历的问题，二维数组是一个图，找到一个图中有几个群

通过bfs解决问题

"""

from collections import deque


class Solution:

    def findCircleNum(self, M) -> int:
        """"""
        q = deque()
        res = 0
        nodes = set(range(len(M)))
        while len(nodes) > 0:
            root = nodes.pop()
            q.appendleft(root)
            while len(q) > 0:
                node = q.pop()
                for i in range(len(M)):
                    if i == node:
                        continue
                    if i in nodes and M[node][i] == 1:
                        q.appendleft(i)
                        nodes.remove(i)
            res += 1
        return res


if __name__ == '__main__':
    m = [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ]
    m = [[1,1,0],
 [1,1,0],
 [0,0,1]]
    r = Solution().findCircleNum(m)
    print('======%s', r)

