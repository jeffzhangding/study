__author__ = 'jeff'

import random
from math import ceil


def heap_sort(l):
    return sorted(l)


def bucket_sort(arr):
    """设置的桶函数为： """
    maximum, minimum = max(arr), min(arr)
    res = []
    bucket_arr = [[] for i in range(int(maximum // 10 - minimum // 10 + 1))]  # set the map rule and apply for space
    for i in arr:  # map every element in array to the corresponding bucket
        index = int(i // 10 - minimum // 10)
        bucket_arr[index].append(i)
    for i in bucket_arr:
        new_bucket = heap_sort(i)   # sort the elements in every bucket
        res.extend(new_bucket)  # move the sorted elements in bucket to array
    return res


if __name__ == '__main__':
    l = [random.uniform(1, 100) for i in range(10000)]
    print(bucket_sort(l))
