__author__ = 'jeff'

"""
给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例：

输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807

"""

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    def copy_list(self, head):
        """"""
        # res = ListNode(head.val)
        # current_node = res
        # while 1:
        #     if head.next is None:
        #         break
        #     current_node.next = ListNode(head.next.val)
        #     current_node, head = current_node.next, head.next
        # return res

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """"""
        last = ListNode((l1.val + l2.val) % 10)
        extral = (l1.val + l2.val) // 10
        res = last
        l1, l2 = l1.next, l2.next
        while 1:
            if l1 is None and l2 is None:
                break
            else:
                num1, num2 = 0, 0
                if l1 is not None:
                    num1 = l1.val
                    l1 = l1.next
                if l2 is not None:
                    num2 = l2.val
                    l2 = l2.next
                last.next = ListNode((num1 + num2 + extral) % 10)
                last = last.next
                extral = (num1 + num2 + extral) // 10

        if extral != 0:
            last.next = ListNode(extral)
        return res


def create_p(l):
    p = ListNode(l[0])
    p_head = p
    for i in l[1:]:
        p.next = ListNode(i)
        p = p.next
    return p_head


def print_list(head):
    while 1:
        if head is None:
            break
        print(head.val)
        head = head.next


if __name__ == '__main__':
    p1 = create_p([2, 4, 3])
    p2 = create_p([5, 6, 4])
    # p1 = create_p([5])
    # p2 = create_p([5])
    # p1 = create_p([1, 8])
    # p2 = create_p([0])
    r = Solution().addTwoNumbers(p1, p2)
    print_list(r)
