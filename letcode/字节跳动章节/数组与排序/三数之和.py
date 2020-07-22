"""
 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。



示例：

给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

"""


class Solution:

    def test(self, nums):
        """"""
        result = list()
        nums_len = len(nums)
        if nums_len < 3:
            return result
        l, r, dif = 0, 0, 0
        nums.sort()
        for i in range(nums_len - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i - 1] == nums[i]:
                continue

            l = i + 1
            r = nums_len - 1
            dif = -nums[i]
            while l < r:
                if nums[l] + nums[r] == dif:
                    result.append([nums[l], nums[r], nums[i]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif nums[l] + nums[r] < dif:
                    l += 1
                else:
                    r -= 1

        return result

    def threeSum(self, nums):
        """"""
        res = []
        s = set()
        for k, v in enumerate(nums):
            if v in s:
                continue
            two_res = self.tow_num(nums[k+1:], -v)
            for i in two_res:
                i.append(v)
                res.append(i)
            if two_res:
                s.add(v)
        return res

    def tow_num(self, numbers, sum):
        """"""
        dt = {}
        res = []
        s = set()
        for k, v in enumerate(numbers):
            if dt.get(sum-v) is not None and v not in s:
                res.append([v, sum-v])
                s.add(v)
            dt[v] = k
        return res

    def remove_repeat(self):
        """"""


if __name__ == '__main__':
    n = [-1, 0, 1, 2, -1, -4]
    r = Solution().threeSum(n)
    # res = Solution().test(n)
    print(r)


