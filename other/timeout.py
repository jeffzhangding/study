"""
超时
"""

import signal
from contextlib import contextmanager
import time


@contextmanager
def timeout(t):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(t)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


def my_func():
    # Add a timeout block.
    with timeout(1):
        print('entering block')

        time.sleep(10)
        print('This should never get printed because the line before timed out')
    print('jeff function == ')


if __name__ == '__main__':
    my_func()
    for i in range(10):
        print('jeff done == %s' % str(i))