__author__ = 'jeff'

"""
 接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

"""


class Solution:

    def trap(self, height) -> int:
        res = 0
        if len(height) == 0:
            return res
        max_index = height.index(max(height))
        left = height[:max_index]
        right = height[max_index+1:]
        max_left = max_index
        while 1:
            if left:
                second_max_left = left.index(max(left))
                res += (max_left-second_max_left-1) * left[second_max_left] - sum(left[second_max_left+1:])
                left = left[:second_max_left]
                max_left = second_max_left
            if right:
                second_max_right = right.index(max(right))
                res += second_max_right * right[second_max_right] - sum(right[:second_max_right])
                right = right[second_max_right+1:]
            if len(left) == 0 and len(right) == 0:
                break
        return res


if __name__ == '__main__':
    lst = [0,1,0,2,1,0,1,3,2,1,2,1]
    print(Solution().trap(lst))


