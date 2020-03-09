"""
无重复字符的最长子串


给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:

输入: "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。


"""

import numpy as np

class Solution:

    def lengthOfLongestSubstring(self, s: str) -> int:
        """"""
        # index_dt = {}
        # for i in range(len(s)):
        #     if s[i] in index_dt.keys():
        #         index_dt[s[i]].append(i)
        #     else:
        #         index_dt[s[i]] = []
        # res = 0
        # for k, v in index_dt.items():
        #     if len(v) == 1:
        #         continue
        #     l_max = np.max(np.array(v[1:]) - np.array(v[:-1]))
        #     if res < l_max:
        #         res = l_max
        # return res
        return self.func2(s)

    def func2(self, s):
        """"""
        if len(s) <= 1:
            return len(s)
        dt = {}
        res = 1
        end_index = len(s) - 1
        for i in range(len(s)):
            reverse_i = len(s) - i - 1
            if s[reverse_i] not in dt.keys():
                dt[s[reverse_i]] = reverse_i
            else:
                sub_l = end_index - reverse_i
                if end_index > dt[s[reverse_i]] - 1:
                    end_index = dt[s[reverse_i]] - 1
                if res < sub_l:
                    res = sub_l
                dt[s[reverse_i]] = reverse_i
        if end_index >= res:
            res = end_index + 1
        return res




if __name__ == '__main__':
    # string = 'bbbbbbbb'
    string = 'abcabcbb'
    # string = " "
    # string = "aa"
    # string = "au"
    string = "abba"

    r = Solution().lengthOfLongestSubstring(string)
    print(r)


