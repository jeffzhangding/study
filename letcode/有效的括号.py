"""
有效的括号
"""

from collections import deque

class Solution:

    def __init__(self):
        # self.k1 = [set(['(', ')']),  set(['{', '}']), set(['[', ']'])]
        self.d = {'(': ')', '[': ']', '{': '}'}
        self.stack = deque()

    def isValid(self, s: str) -> bool:
        if not s:
            return True
        if len(s) % 2 != 0:
            return False
        for i in s:
            if i in self.d.keys():
                self.stack.append(i)
            else:
                if len(self.stack) == 0:
                    return False
                elif self.d.get(self.stack[-1]) == i:
                    self.stack.pop()
                else:
                    return False
        return len(self.stack) == 0

    def get_subtring(self, s):
        if self.check(s[0], s[-1]):
            return s[1:-1]
        elif self.check(s[0], s[1]):
            return s[2:]
        else:
            return False

    def check(self, word1, word2):
        """"""
        k = self.d.get(word1)
        if k and k == word2:
            return True
        else:
            return False


if __name__ == '__main__':
    s = '()[]{}'
    s = "(]"
    s = '{[]}'
    s = '){'
    res = Solution().isValid(s)
    print(res)


