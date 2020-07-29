__author__ = 'jeff'

"""
给出一个区间的集合，请合并所有重叠的区间。

示例 1:

输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2:

输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

"""

class Solution:

    def merge(self, intervals):
        if len(intervals) == 0:
            return []
        intervals.sort(key=lambda m: m[0])
        res = []
        r = intervals[0]
        for i in range(1, len(intervals)):
            if r[1] >= intervals[i][0]:
                r[1] = max(r[1], intervals[i][1])
            else:
                res.append(r)
                r = intervals[i]
        res.append(r)
        return res


if __name__ == '__main__':
    # l = [[1,3],[2,6],[8,10],[15,18]]
    l =  [[1,4],[4,5]]
    # l = [[1,4],[0,2],[3,5]]
    print(Solution().merge(l))




