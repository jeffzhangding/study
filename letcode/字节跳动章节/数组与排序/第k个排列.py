__author__ = 'jeff'

"""
给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

说明：

给定 n 的范围是 [1, 9]。
给定 k 的范围是[1,  n!]。
示例 1:

输入: n = 3, k = 3
输出: "213"
示例 2:

输入: n = 4, k = 9
输出: "2314"
"""

import math


class Solution:

    def getPermutation(self, n: int, k: int) -> str:
        res = []
        n_list = list(range(1, n+1))
        math.factorial(n)
        current_k = k
        for i in range(1, n+1):
            m = math.factorial(n-i)
            x = current_k // m
            b = current_k % m
            if b == 0:
                x -= 1
                b = m
            current_k = b
            res.append(str(n_list[x]))
            n_list.pop(x)
        return ''.join(res)


if __name__ == '__main__':
    _n, _k = 4, 12
    r = Solution().getPermutation(_n, _k)
    print('=========%s', r)

