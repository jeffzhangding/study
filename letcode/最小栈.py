"""
最小栈
"""

from collections import deque
from math import floor


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        #self.stack = deque()
        self.stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        #return min(self.stack)
        return self.min_value(self.stack)

    def min_value(self, l):
        """"""
        lenth = len(l)
        if lenth == 1:
            return l[0]
        elif lenth == 2:
            return min(l)
        elif lenth == 0:
            return None
        else:
            n = floor(lenth / 2.0)
            number1 = self.min_value(l[:n])
            number2 = self.min_value(l[n:])
            if number1 >= number2:
                return number2
            else:
                return number1


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

if __name__ == '__main__':
    obj = MinStack()
    import random
    m = [random.randint(1, 10000) for i in range(1000) ]
    # for i in range(1000):
    print(m)
    res = obj.min_value(m)
    print('obj: %s==== p: %s' % (res, min(m)))


