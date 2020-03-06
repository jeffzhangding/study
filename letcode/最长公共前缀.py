__author__ = 'jeff'

"""
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

示例 1:

输入: ["flower","flow","flight"]
输出: "fl"
示例 2:

输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
说明:

所有输入只包含小写字母 a-z 。

解决思路：
    使用前缀树来解决

"""

class Solution:
    def longestCommonPrefix(self, strs):
        """"""
        # 构建前缀树
        root = {'key': '', 'son': {}, 'times': len(strs)}
        for s in strs:
            if s == '':
                return ''
            father = root
            for i in s:
                if i in father['son'].keys():
                    father['son'][i]['times'] += 1
                else:
                    father['son'][i] = {'key': i, 'son': {}, 'times': 1}
                father = father['son'][i]

        # 搜索前缀树获取最大公共子串
        res = []
        f = root
        while True:
            if f['times'] < len(strs):
                break
            res.append(f['key'])
            if len(f['son']) == 0:
                break
            else:
                f = list(f['son'].values())[0]
        return ''.join(res)


    # def get_max_prefix(self, root, max_len):
    #     """搜索前缀树"""


if __name__ == '__main__':

    st = ["dog", "racecar", "car"]
    # st = ["flower","flow","flight"]
    # st = ["","a"]
    st = ["aa","a"]

    res = Solution().longestCommonPrefix(st)
    print(res)



# 别人的代码（看着还不错，只是这个问题的话）
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        max = ''
        for i in zip(*strs):
            if len(set(i)) == 1:
                max += i[0]
            else:
                return max
        return max

