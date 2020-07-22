__author__ = 'jeff'

"""
在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
"""


class Solution:

    def findKthLargest(self, nums, k: int) -> int:
        """"""
        partion_nums = nums
        partion_k = k

        while 1:
            number = partion_nums[partion_k-1]
            left = []
            right = []
            middel = []
            for i in partion_nums:
                if i == number:
                    middel.append(i)
                elif i > number:
                    left.append(i)
                else:
                    right.append(i)
            if len(left) >= partion_k:
                partion_nums = left
            elif partion_k - len(left) <= len(middel):
                return number
            else:
                partion_nums = right
                partion_k = partion_k - len(left) - len(middel)






    # def partion(self, nums, k):
    #     """"""
    #     if len(nums)



if __name__ == '__main__':
    l = [3,2,1,5,6,4]
    l = [1, 2, 2, 3, 1, 1, 1]
    l = [3,1,2,4]
    k = 2
    r = Solution().findKthLargest(l, k)
    print('=======%s', r)
