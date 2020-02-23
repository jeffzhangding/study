__author__ = 'jeff'

"""
 逆波兰表达式求值
"""

from collections import deque
from math import floor, ceil

class Solution:

    def evalRPN(self, tokens) -> int:
        """"""
        stack = deque()
        words = {'+': 'add', '-': 'sub', '*':  'multi', '/': 'div'}
        for word in tokens:
            if word in words.keys():
                right = int(stack.pop())
                left = int(stack.pop())
                r = getattr(self, words[word])(left, right)
                stack.append(r)
            else:
                stack.append(word)
        return stack.pop()

    def add(self, a, b):
        return a + b

    def div(self, a, b):
        r = a / b
        if r < 0:
            return ceil(r)
        else:
            return floor(r)

    def sub(self, a, b):
        return a - b

    def multi(self, a, b):
        return a * b


if __name__ == '__main__':
    k = ["4", "13", "5", "/", "+"]
    k = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    k = ["4", "-2", "/", "2", "-3", "-", "-"]
    res = Solution().evalRPN(k)
    print(res)


