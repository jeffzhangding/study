__author__ = 'jeff'

from functools import wraps
from datetime import datetime
import time
import numpy as np


def debug_method(name):

    def mm(fn):
        # @wraps(fn)
        def fun(*args, **kwargs):
            # print(name)
            t1 = datetime.now()
            res = fn(*args, **kwargs)
            t2 = datetime.now()
            # print('=====funtion: %s,  timing: %s' % (fn, str(t2-t1)))
            return res

        return fun
    return mm


@debug_method('123')
def test1(debug_name='kg'):
    time.sleep(0.1)
#
#
# @debug_method('456')
# def test2():
#     time.sleep(0.2)

def say_hello(contry):
    def wrapper(func):
        def deco(*args, **kwargs):
            if contry == "china":
                print("你好!")
            elif contry == "america":
                print('hello.')
            else:
                return

            # 真正执行函数的地方
            func(*args, **kwargs)

        return deco

    return wrapper


@say_hello("china")
def chinese():
    print("我来自中国。")


@say_hello("america")
def american():
    print("I am from America.")


def test2():
    """"""
    a = np.array(
        [
            [[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]
        ]
    )
    a = np.array(
        [
            [[1, 2], [3, 4]], [[5, 6], [7, 8]]
        ]
    )
    b = np.array(
        [
            [1, 2], [3, 4]
        ]
    )
    c = np.array(
        [
            [1, 2], [5, 6]
        ]
    )
    # print(np.dot(b, b))
    # print(np.multiply(b, c))
    #
    print(a.T)
    res = np.dot(a, a)
    # print(res)
    print(a * a.T)


if __name__ == '__main__':
    # chinese()
    # chinese()
    # test1()
    # test1()
    test2()

