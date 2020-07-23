__author__ = 'jeff'

"""
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
"""


class Solution:
    def longestConsecutive(self, nums) -> int:
        res = 0
        set_list = set(nums)
        while len(set_list) > 0:
            current_lenth = 1
            e = set_list.pop()
            back_nx_e = e
            foward_e = e
            while 1:
                back_nx_e += 1
                if back_nx_e in set_list:
                    current_lenth += 1
                    set_list.remove(back_nx_e)
                else:
                    break
            while 1:
                foward_e -= 1
                if foward_e in set_list:
                    current_lenth += 1
                    set_list.remove(foward_e)
                else:
                    break
            res = max(res, current_lenth)
        return res


if __name__ == '__main__':
    l = [100, 4, 200, 1, 3, 2]
    r = Solution().longestConsecutive(l)
    print('======%s', r)

