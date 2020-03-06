"""
目标和

思路：
问题是一个遍历问题，理论最大次数为 2 的 n次方

两个优化方案
1.线性规划：每次向下遍历记录上一次的所有结果，可以去除重复计算
2、前后同时遍历，然后两个集合组合，组合之前可以去除不可能的情况，从而减少组合次数
"""

from math import ceil


class Solution:

    def findTargetSumWays(self, nums, S) -> int:
        """"""
        # res = 0
        l = len(nums)
        # if len(nums) == 1:
        #     if nums[0] == S:
        #         res += 1
        #     if -nums[0] == S:
        #         res += 1
        #     return res
        if l < 4:
            dt = self.get_number_dict(nums)
            res = 0
            for k, v in dt.items():
                if k == S:
                    res += v
            return res

        m = ceil(l / 2.0)
        dt1 = self.get_number_dict(nums[:m])
        dt2 = self.get_number_dict(nums[m:])
        res = self.merge(dt1, dt2, S)
        return res

    def get_number_dict(self, l):
        """"""
        dt = {}
        dt[l[0]] = dt.get(l[0], 0) + 1
        dt[-l[0]] = dt.get(-l[0], 0) + 1

        for i in l[1:]:
            d = {}
            for k, v in dt.items():
                new_k1 = k + i
                new_k2 = k - i
                d[new_k1] = d.get(new_k1, 0) + v
                d[new_k2] = d.get(new_k2, 0) + v
            dt = d
        return dt

    def merge(self, dt1, dt2, s):
        """"""
        res = 0
        new_dt = {}
        max_v = max(dt1.keys())
        min_v = min(dt1.keys())
        for k2, v2 in dt2.items():
            if min_v + k2 <= s <= max_v + k2:
                new_dt[k2] = v2

        for k, v in dt1.items():
            for k3, v3 in new_dt.items():
                if k + k3 == s:
                    res += v * v3
        return res


def f2():
    """背包问题"""



if __name__ == '__main__':
    s = 1
    # n = [1, 1, 1, 1, 1]
    n = [0,0,0,0,0,0,0,0,1]
    res = Solution().findTargetSumWays(n, s)
    print(res)


