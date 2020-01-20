__author__ = 'jeff'


class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.q = [None for i in range(k + 1)]
        self.head = 0
        self.tail = 0
        self.k = k

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if self.isFull():
            return False
        else:
            self.tail = self.next_index(self.tail)
            self.q[self.tail] = value
            return True

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.isEmpty():
            return False
        else:
            self.head = self.next_index(self.head)
            return True

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            next_head = self.next_index(self.head)
            return self.q[next_head]

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            res = self.q[self.tail]
            # self.tail = self.last_index(self.tail)
            return res

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return self.head == self.tail

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return self.next_index(self.tail) == self.head

    def next_index(self, index):
        """
        """
        if index == self.k:
            return 0
        else:
            return index + 1

    def last_index(self, index):
        if index == 0:
            return self.k
        else:
            return index - 1


if __name__ == '__main__':
    k = 5
    obj = MyCircularQueue(0)
    print(obj.isEmpty())
    print(obj.isFull())
    for i in range(k):
        print(obj.enQueue(i))
    print(obj.isFull())
    print(obj.Front())
    print(obj.Rear())

    # print(obj.deQueue())