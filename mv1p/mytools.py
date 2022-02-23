import time
from functools import wraps
import numpy as np

timer_calc={}

def fn_timer(function):
    
    global timer_calc
    if function.__name__ not in timer_calc:
        timer_calc[function.__name__]=0

    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        timer_calc[function.__name__]+=t1-t0
        return result
    return function_timer

def time_print():
    for fn,excuteTime in timer_calc.items():
        print("Total time running %s: %s seconds" %
            (fn, str(excuteTime))
            )

def normal(x):
    return x/np.linalg.norm(x)


def SplitP(P):
    if P is None:
        return None,None
    return P[:,:3],P[:,3]
def MergeRT(R,T):
    if R is None or T is  None:
        return  None
    return np.hstack((R,T.reshape(3,1)))