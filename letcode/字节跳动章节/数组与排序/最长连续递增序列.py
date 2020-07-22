__author__ = 'jeff'

"""
输入: [1,3,5,4,7]
输出: 3
解释: 最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为5和7在原数组里被4隔开。 
"""


class Solution:

    def findLengthOfLCIS(self, nums) -> int:
        """"""
        max_lenth = 0
        current_lenth = 0
        last_number = None
        for i in nums:
            if last_number is None:
                current_lenth += 1
                max_lenth = current_lenth
                last_number = i
                continue
            if i > last_number:
                current_lenth += 1
            else:
                current_lenth = 1
            last_number = i
            if current_lenth > max_lenth:
                max_lenth = current_lenth
        return max_lenth


if __name__ == '__main__':
    l = [1,3,5,4,7]
    # l = [2,2,2,2,2]
    res = Solution().findLengthOfLCIS(l)
    print('=====%s', res)


