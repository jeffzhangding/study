__author__ = 'jeff'

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# matplotlib.use('Qt5Agg')


def sample_example():
    """"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.plot([1, 2, 3, 4], [1, 2, 3, 3])
    # plt.show()


# plt.subplots


if __name__ == '__main__':

    sample_example()
    # plt.savefig('test.png')
    print('==')



