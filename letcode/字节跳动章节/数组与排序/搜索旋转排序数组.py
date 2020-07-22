__author__ = 'jeff'


class Solution:
    def search(self, nums, target):
        """"""
        # r = self.dichotomy(nums, target, m=len(nums)-1)
        if len(nums) == 0:
            return -1
        spin_index = self.dichotomy_spin(nums, len(nums)-1)
        r = -1
        if spin_index is None:
            return self.dichotomy_search_index(nums, target, len(nums)-1)
        if nums[spin_index] <= target <= nums[-1]:
            r = self.dichotomy_search_index(nums[spin_index:], target, len(nums) - 1)
        elif nums[0] <= target <= nums[spin_index - 1]:
            r = self.dichotomy_search_index(nums[:spin_index], target, spin_index - 1)
        return r

    def dichotomy_spin(self, nums, m):
        """递归二分"""
        if nums[0] < nums[-1]:
            return None
        if len(nums) <= 2:
            return m
        else:
            med = len(nums) // 2
            b = len(nums) % 2
            med_index = med + b
            if nums[0] < nums[med_index-1] and nums[med_index] < nums[-1]:
                return m - med + 1
            elif nums[0] < nums[med_index-1]:
                return self.dichotomy_spin(nums[med_index:], m)
            else:
                return self.dichotomy_spin(nums[:med_index], m-med)

    def dichotomy_search_index(self, nums, target, m):
        """"""
        if len(nums) == 0:
            return -1
        elif len(nums) == 1:
            if nums[0] == target:
                return m
            else:
                return -1
        else:
            # med_index = len(nums) // 2
            # b = len(nums) % 2
            # curent_index = m - med_index - b
            med = len(nums) // 2
            b = len(nums) % 2
            med_index = med + b
            # if nums[med_index-1] == target:
            #     return m - med
            if nums[med_index] > target:
                right = nums[:med_index]
                return self.dichotomy_search_index(right, target, m-med)
            else:
                return self.dichotomy_search_index(nums[med_index:], target, m)

    def dichotomy(self, nums, target, m):
        """递归二分"""
        if len(nums) == 0:
            return -1
        elif len(nums) == 1:
            if nums[0] == target:
                return m
            else:
                return -1
        elif len(nums) == 2:
            if target == nums[1]:
                return m
            elif target == nums[0]:
                return m - 1
            else:
                return -1
        else:
            med = len(nums) // 2
            b = len(nums) % 2
            med_index = med + b
            # curent_index = m - med_index - b
            if nums[0] < nums[med_index-1]:
                if nums[0] <= target <= nums[med_index-1]:
                    return self.dichotomy_search_index(nums[:med_index], target, m-med)
                else:
                    right = nums[med_index:]
                    return self.dichotomy(right, target, m)
            else:
                return self.dichotomy(nums[:med_index], target, m-med)


    def iter_dichotomy(self):
        """迭代二分法"""


if __name__ == '__main__':
    n = [4, 5, 6, 7, 0, 1, 2]
    # n = [1, 3, 5]
    # n = [5, 1, 3]
    tag = 0
    res = Solution().search(n, tag)
    print('=====res: %s', res)

