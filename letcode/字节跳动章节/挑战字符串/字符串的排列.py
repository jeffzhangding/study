__author__ = 'jeff'

"""
字符串的排列

给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的子串。

示例1:

输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
 

示例2:

输入: s1= "ab" s2 = "eidboaoo"
输出: False
 

注意：

输入的字符串只包含小写字母
两个字符串的长度都在 [1, 10,000] 之间

思路：
每次截取与匹配字符串固定长度的字符，判断每种字符的个数一样，便是在排列中

"""

import collections

class Solution:

    def checkInclusion(self, s1: str, s2: str) -> bool:
        """"""
        return self.fun2(s1, s2)
        # if len(s1) > len(s2):
        #     return False
        # for i in range(len(s2) + 1 - len(s1)):
        #     sub_s2 = s2[i: i + len(s1)]
        #     if collections.Counter(sub_s2) == collections.Counter(s1):
        #         return True
        # return False

    def fun2(self, s1, s2):
        """"""
        if len(s1) > len(s2):
            return False

        count_s1 = collections.Counter(s1)
        first_count = collections.Counter(s2[:len(s1)])
        dt = {}
        for k in list(count_s1.keys()) + list(first_count.keys()):
            new_v = first_count.get(k, 0) - count_s1.get(k, 0)
            if new_v != 0:
                dt[k] = new_v
        if len(dt) == 0:
            return True
        for i in range(len(s2) - len(s1)):
            key = s2[i+len(s1)]
            diff_v = dt.get(key, 0) + 1
            if diff_v == 0:
                dt.pop(key)
            else:
                dt[key] = diff_v
            last_diff_v = dt.get(s2[i], 0) - 1
            if last_diff_v == 0:
                dt.pop(s2[i])
            else:
                dt[s2[i]] = last_diff_v
            if len(dt) == 0:
                return True

        return False



if __name__ == '__main__':
    # s1 = "ab"
    # s2 = "eidbaooo"

    s1 = "ab"
    s2 = "eidboaoo"

    # s1 = "adc"
    # s2 = "dcda"
    s1 = "abc"
    s2 = "cccccbabbbaaaa"

    res = Solution().checkInclusion(s1, s2)
    print(res)

