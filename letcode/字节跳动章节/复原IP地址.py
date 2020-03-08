"""
给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。

示例:

输入: "25525511135"
输出: ["255.255.11.135", "255.255.111.35"]

"""

from collections import deque

class Solution:
    def restoreIpAddresses(self, s: str):
        """"""
        res_list = self.get_sub_string(s, sub_number=4)
        res = []
        for i in res_list:
            res.append('.'.join(i))
        return res

    def get_sub_string(self, s, sub_number=4):
        """"""
        if len(s) > 3*sub_number or sub_number < 1 or len(s) < sub_number:
            return []
        if sub_number == 1:
            if len(str(int(s))) != len(s) or int(s) > 255:
                return []
            return [deque([s])]
        res = deque()
        for i in range(3):
            if i + (sub_number-1) > len(s):
                continue
            elif int(s[:i+1]) > 255:
                continue
            elif len(str(int(s[:i+1]))) != i+1:
                continue
            else:
                suffix_list = self.get_sub_string(s[i+1:], sub_number-1)
                for suf in suffix_list:
                    suf.appendleft(s[:i+1])
                res.extend(suffix_list)

        return res


if __name__ == '__main__':

    st = '25525511135'
    st = "010010"

    res = Solution().restoreIpAddresses(st)
    print(res)

