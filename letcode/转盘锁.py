"""
转盘锁问题
"""

from datetime import datetime
from queue import Queue

def debug_method(fn):

    def fun(*args, **kwargs):
        t = datetime.now()
        res = fn(*args, **kwargs)
        print('===== %s '  % str(datetime.now() - t))
        return res

    return fun


class Solution(object):

    # def __new__(cls, *args, **kwargs):
    #     cls.nodes, cls.relations = cls.generate_graph()
    #     return object.__new__(cls)

    # def __init__(self):
    #     super(Solution, self).__init__()


    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        if target == '0000':
            return 0
        if '0000' in deadends:
            return -1
        dead_nodes = set(deadends)
        # bf_tree = self._bfs('0000', target, dead_nodes)
        bf_tree = self.get_bfs_tree('0000', target, dead_nodes)
        if target not in bf_tree.keys():
            return -1
        else:
            return self.get_deep(bf_tree, target)

    def get_deep(self, bf_tree, target):
        """"""
        deep = 1
        father = bf_tree[target]
        debug = [target, father]
        while deep <= len(bf_tree):
            if father == '0000':
                break
            father = bf_tree[father]
            debug.append(father)
            deep += 1
        # print(debug)
        return deep

    @classmethod
    def generate_graph(cls):
        """构建图"""
        nodes = set(cls.get_nodes())
        relations = {}
        for n in nodes:
            r = set()
            for i in range(4):
                number = int(n[i])
                n1, n2 = list(n), list(n)
                n1[i], n2[i] = cls.get_next_number(number)
                n1, n2 = ''.join(n1), ''.join(n2)
                r.add(n1)
                r.add(n2)
                # if n1 not in dead_nodes:
                #     r.add(n1)
                # if n2 not in dead_nodes:
                #     r.add(n2)
            relations[n] = r
        return nodes, relations

    @classmethod
    def get_nodes(cls):
        """获取所有节点字符串"""
        res = []
        for i in range(10000):
            n = str(i)
            l = len(str(i))
            if l < 4:
                n_list = []
                for k in range(4-l):
                    n_list.append('0')
                n_list.append(n)
                n = ''.join(n_list)
            res.append(n)
        return res

    @classmethod
    def get_next_number(cls, number):
        """"""
        return str((number + 1) % 10), str((number+9) % 10)

    def _bfs(self, start_node, target=None, dead_nodes=None):
        """广度优先遍历， 返回层数"""
        bf_tree = {}
        q = Queue()
        q.put(start_node)
        walk_book = set()
        bf_tree[start_node] = start_node
        while not q.empty():
            node = q.get()
            if dead_nodes and node in dead_nodes:
                continue
            if node in walk_book:
                continue
            else:
                walk_book.add(node)
            for n in self.relations[node]:
                if dead_nodes and node in dead_nodes:
                    continue

                if n in walk_book:
                    continue
                bf_tree[n] = node
                q.put(n)
                if target and n == target:
                    return bf_tree

        return bf_tree

    def get_bfs_tree(self, start_node, target=None, dead_nodes=None):
        """"""
        bf_tree = {}
        q = Queue()
        q.put(start_node)
        walk_book = set()
        bf_tree[start_node] = start_node
        while not q.empty():
            node = q.get()
            if dead_nodes and node in dead_nodes:
                continue
            if node in walk_book:
                continue
            else:
                walk_book.add(node)

            n_list = []

            for i in range(4):
                number = int(node[i])
                n1, n2 = list(node), list(node)
                n1[i], n2[i] = self.get_next_number(number)
                n1, n2 = ''.join(n1), ''.join(n2)
                n_list.append(n1)
                n_list.append(n2)

            for n in n_list:
                if dead_nodes and node in dead_nodes:
                    continue

                if n in walk_book:
                    continue
                bf_tree[n] = node
                q.put(n)
                if target and n == target:
                    return bf_tree

        return bf_tree


if __name__ == '__main__':
    deadends = ["0201", "0101", "0102", "1212", "2002"]
    target = "0202"

    # deadends = ["8888"]
    # target = "0009"

    # deadends = ["8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888"]
    # target = "8888"
    #
    deadends = ["2123", "3220", "0001", "0310", "1332", "0123", "1110", "0311", "1303", "1221"]
    target = "1011"

    deadends = ["6586", "6557", "0399", "3436", "1106", "4255", "1161", "7546", "2375", "5535", "7623", "0805", "7045", "8244",
     "1804", "1777", "5152", "7241", "4488", "3653", "7485", "9103", "2726", "4624", "8654", "1404", "9321", "5145",
     "4237", "5423", "9350", "3383", "8658", "2601", "2446", "1605", "6804", "1521", "0832", "5555", "6710", "3851",
     "6370", "0069", "7369", "6352", "4165", "4327", "9727", "1363", "9906", "9463", "8628", "5239", "0009", "2743",
     "0419", "4722", "7251", "5645", "5159", "4040", "1406", "5836", "0623", "9851", "2970", "0479", "1707", "5248",
     "0135", "8840", "9395", "1068", "9653", "4461", "6830", "7851", "7798", "3745", "1608", "2061", "5404", "3536",
     "3875", "3552", "8430", "0846", "5575", "2835", "1777", "5848", "5181", "8129", "2408", "3257", "9168", "3279",
     "4705", "9799", "1592", "7849", "4934", "1210", "0384", "3946", "5200", "3702", "4792", "1363", "0340", "4623",
     "9837", "0798", "2400", "0859", "3002", "1819", "2925", "8966", "7065", "3310", "1415", "9986", "7612", "1233",
     "9681", "6869", "5324", "4271", "1632", "2947", "8829", "9102", "9502", "4896", "2556", "4998", "7642", "8477",
     "4439", "8391", "7171", "2081", "5401", "0369", "4498", "1269", "2535", "7805", "6611", "1605", "1432", "6237",
     "5565", "9618", "2123", "5178", "3649", "8657", "6236", "6737", "1561", "1802", "1349", "9738", "6245", "7202",
     "8442", "7183", "5105", "7963", "0259", "5622", "3098", "0664", "7366", "1556", "5711", "9981", "4607", "2063",
     "7540", "1818", "7320", "8505", "1028", "6127", "1816", "8961", "7126", "4739", "4050", "7729", "5887", "4836",
     "1244", "2697", "3937", "9817", "2759", "9536", "0154", "7214", "5688", "1284", "5434", "7103", "2704", "6790",
     "3244", "8797", "3860", "1988", "1458", "4268", "1901", "4787", "7599", "6672", "3579", "3726", "6670", "1603",
     "3332", "7249", "0984", "6783", "4456", "0023", "2678", "0167", "8626", "6080", "5716", "5083", "6135", "8700",
     "7890", "8683", "2089", "0264", "2123", "0787", "3056", "2647", "4645", "8748", "6936", "6899", "0031", "4934",
     "0221", "9481", "9959", "1386", "7695", "2034", "0466", "0809", "9166", "6381", "6937", "0744", "8059", "8498",
     "5772", "8379", "4448", "5794", "7423", "2568", "4671", "6408", "4335", "1655", "3662", "1250", "5262", "7197",
     "6831", "8004", "0575", "8784", "2920", "0869", "7157", "0153", "7255", "1541", "1247", "5498", "0566", "6632",
     "7640", "1733", "2546", "5110", "2852", "8042", "8175", "0284", "8589", "8918", "5755", "2289", "0812", "4850",
     "4650", "9018", "6649", "5099", "6532", "9891", "8675", "1718", "5442", "6786", "8915", "3710", "3833", "2659",
     "7040", "3959", "2505", "7574", "1199", "3465", "4557", "7230", "9161", "5177", "7815", "4564", "1470", "8051",
     "6287", "2504", "4025", "8911", "6158", "6857", "8948", "7991", "3670", "3413", "0423", "7184", "7921", "1351",
     "8908", "1921", "1685", "5579", "4641", "0286", "6410", "2800", "7018", "1402", "7410", "3471", "1312", "9530",
     "4581", "5364", "4820", "8192", "3088", "4714", "2255", "2342", "5042", "8673", "9788", "2203", "0879", "2345",
     "9712", "2008", "0652", "0939", "0720", "2954", "4482", "2390", "0807", "4634", "6266", "5222", "6898", "7491",
     "0294", "1811", "0667", "8282", "5754", "1841", "9518", "9093", "7904", "4902", "0068", "5157", "7823", "8073",
     "8801", "8179", "1402", "9977", "2332", "9448", "2251", "8455", "6157", "1878", "4183", "3331", "8047", "1254",
     "9639", "2156", "5780", "7359", "0260", "9683", "6842", "1098", "6495", "2057", "6583", "0932", "2577", "1818",
     "6042", "8358", "1833", "5512", "4529", "0583", "9955", "9205", "6055", "3322", "2232", "5372", "5835", "2202",
     "9696", "1596", "3424", "3696", "5695", "1365", "6432", "0327", "1565", "8509", "6936", "3363", "3007", "3107",
     "0410", "6258", "2492", "0300", "1255", "1664", "8666", "6826", "9961", "5782", "0140", "5567", "9596", "1680",
     "1892", "5016", "8804", "4962", "9318", "4540", "5044", "0979", "2004", "4265", "7689", "0289", "3434", "6090",
     "1375", "3135", "3935", "5140", "9255", "3997", "3482", "8150", "8164", "0787"]
    target = "8828"

    t = datetime.now()
    res = Solution().openLock(deadends, target)
    print(datetime.now() - t)
    print(res)



