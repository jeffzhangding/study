"""
 字符串相乘
给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

示例 1:

输入: num1 = "2", num2 = "3"
输出: "6"
示例 2:

输入: num1 = "123", num2 = "456"
输出: "56088"
说明：

num1 和 num2 的长度小于110。
num1 和 num2 只包含数字 0-9。
num1 和 num2 均不以零开头，除非是数字 0 本身。
不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。

思路：
1.还是要将字符串转换成int类型，然后相乘

2.根据乘法规则处理


"""

from collections import deque
# import numpy as np


class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        """"""
        return self.fun2(num1, num2)
        # return str(self.trans_to_number(num1) * self.trans_to_number(num2))

    def trans_to_number(self, s):
        """"""
        dt = dict(zip(list('0123456789'), list(range(10))))
        if s[0] == '0':
            return 0
        res = 0
        l = len(s)
        for i in range(l):
            res += dt[s[i]] * pow(10, l - i - 1)
        return res

    def fun2(self, num1, num2):
        """"""
        dt = dict(zip(list('0123456789'), list(range(10))))
        res = deque()
        m = deque()
        for i in range(len(num1)):
            k = deque()
            for j in range(len(num2)):
                r = dt[num1[len(num1) - i -1]] * dt[num2[len(num2) - j -1]]
                if len(k) == 0:
                    k.appendleft(r % 10)
                    k.appendleft(r // 10)
                else:
                    add = k[0] + r
                    k[0] = add % 10
                    k.appendleft(add // 10)
            m.append(k)
            add_res = 0
            for history_k in range(len(m)):
                if len(m[history_k]) == 0:
                    # pop_list.append(history_k)
                    continue
                add_res += m[history_k].pop()
            if add_res > 9:
                m.appendleft(deque([add_res // 10]))
            res.appendleft(str(add_res % 10))

        for l in range(len(num1) + len(num2) - len(res)):
            add_res = 0
            # pop_list = []
            for history_k in range(len(m)):
                if len(m[history_k]) == 0:
                    # pop_list.append(history_k)
                    continue
                add_res += m[history_k].pop()
            if add_res > 9:
                m.appendleft(deque([add_res // 10]))
            res.appendleft(str(add_res % 10))
        for i in range(len(res)):
            if res[i] != '0':
                return ''.join(list(res)[i:])
        return '0'










if __name__ == '__main__':
    num1 = "123"
    num2 = "456"
    num1, num2 = '456',  '123'

    res = Solution().multiply(num1, num2)
    print(res)

