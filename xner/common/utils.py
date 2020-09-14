import time
import sys
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kw):
        func_name = ""
        try:
            func_name = func.__qualname__
        except:
            # TODO: how to get __qualname__ of staticmethod function?
            # (__name__ is not enough, because it dont have class name)
            func_name = func.__name__
        sys.stdout.write(f"[{func_name}]: running ...")
        start_time = time.time()
        ret = func(*args, **kw)
        sys.stdout.flush()
        sys.stdout.write(f"\r[{func_name}]: run time is {round(time.time() - start_time, 3)} s\n")
        return ret

    return wrapper
