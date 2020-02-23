"""
完全平方数问题
"""

from math import floor, sqrt


class Solution:
    def numSquares(self, n: int) -> int:
        max_num = floor(sqrt(n))
        number_list = [pow(i, 2) for i in list(range(1, max_num+1))]
        number_set = set(number_list)
        if n in number_set:
            return 1
        for i in range(2, n+1):
            s = set()
            for k in number_set:
                for number in number_list:
                    new_n = k + number
                    if new_n == n:
                        return i
                    elif new_n < n:
                        s.add(new_n)
                    else:
                        break
            number_set = s


if __name__ == '__main__':
    res = Solution().numSquares(12)
    print(res)

