"""
每日温度
总结：
在推理过程中，逆序遍历的时候，逻辑推理错误，得出正反遍历复杂度都一样的结论，所以最后没有完成
推理过程不能下意识的，应该仔细的思考
"""

import numpy as np
from collections import deque

class Solution:
    def dailyTemperatures(self, T):
        if len(T) == 1:
            return [0]
        # return self.f1(T)
        return self.f4(T)

    def f1(self, T):
        res = [0] * len(T)
        t = np.array(T)
        t2 = t[1:] - t[:-1]
        index = np.array(list(range(1, len(t))))
        t_index = index[t2 > 0]
        head = 0
        for i in range(len(T)):
            for k in t_index[head:]:
                if T[i] < T[k]:
                    res[i] = k - i
                    break
            if head < len(t_index) and t_index[head] <= i:
                head += 1
        return res

    def f2(self, t):
        """"""
        res = [0] * 10
        stack = deque()
        stack.append(t[0])
        for i in range(1, len(t)):
            if t[i] > t[stack[-1]]:
                for k in stack[:-1]:
                    pass

    def f3(self, t):
        bucket = [[]] * 71
        for i in range(len(t)):
            bucket[t[i] - 30].append(i)

    def f4(self, t):
        """逆序遍历"""
        res = [0] * len(t)
        index = list(range(len(t)))
        index.reverse()
        stack = deque()
        for i in index:
            if len(stack) == 0:
                stack.append(i)
            else:
                while len(stack) > 0:
                    if t[stack[-1]] > t[i]:
                        res[i] = stack[-1] - i
                        break
                    else:
                        stack.pop()
                stack.append(i)

        return res

def test():
    """"""
    t = [73, 74, 75, 71, 69, 72, 76, 73]
    k = list(zip(t, list(range(8))))
    k.sort(key=lambda x: x[0])
    print(k)

if __name__ == '__main__':
    t = [73, 74, 75, 71, 69, 72, 76, 73]
    res = Solution().dailyTemperatures(t)
    print(res)
    # print(test())
