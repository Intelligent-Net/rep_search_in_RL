import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc as tm
import psutil
from resource import *

class Stats:
    def __init__(self):
        self.tic = time.perf_counter()
        tm.start()
        self.current, _ = tm.get_traced_memory()

    def end(self):
        _, peak = tm.get_traced_memory()
        cpu = getrusage(RUSAGE_SELF)[0]
        tim = time.perf_counter() - self.tic
        mem = peak - self.current
        print(f"Elapsed Time: {tim:0.4f} secs, Memory Used: {mem / 10**3} KB, CPU {cpu:0.4f}")

def argmax(x):
    # numpy argmax replacement as it chooses lowest index when duplicates
    x = list(x)
    mx = max(x)
    if x.count(mx) > 1:
        return np.random.choice([i for i in range(len(x)) if x[i] == mx])
    else:
        return x.index(mx)

def softmax(x, tau=1.0):
    # softmax(d)
    #x = x / tau
    # Subtract max for numeric stability
    n = np.exp(x - np.max(x))
    # Take care of both 1 and 2 dimensional cases
    #dist = n / np.sum(n) if n.ndim == 1 else n / np.sum(n, 1, keepdims=True)
    dist = n / np.sum(n, n.ndim - 1, keepdims=True)

    # biased random choice
    return np.random.choice(np.arange(len(dist)), size=1, p=dist)[0]

def scale(from_min, from_max, to_min, to_max, value):
    def lscale(v):
        return to_min + ((v - from_min) * (to_max - to_min)) / (from_max - from_min)

    if type(value) is float:
        return lscale(value)
    else:
        return [lscale(v) for v in value]

def ndiv(n, fr, to):
    if n == 1:
        return [(fr + to) / 2.0]
    b = (to - fr) / (n + 2)
    return scale(0, n, fr - b, to + b, [i for i in range(1, n)])

def bound(n, fr, to):
    return [ndiv(i, j, k) for i, j, k in zip(n, fr, to)]

def set_range(r, os):
    return [os[i] if v == -1 else v for i, v in enumerate(r)] 

def discretise2(buckets, vs, lb, ub):
    ratios = [(o + abs(lb[i])) / (ub[i] - lb[i]) for i, o in enumerate(vs)]
    nobs = [int(round((buckets[i] - 1) * r)) for i, r in enumerate(ratios)]
    return tuple([min(buckets[i] - 1, max(0, o)) for i, o in enumerate(nobs)])

def discretise3(bounds, vs):
    def in_bound(bound, v):
       l = len(bound)
       if l == 1 or v <= bound[0]:
            return 0
       #elif v > bound[l - 1]:
       #     return l
       else:
            #print(v, bound)
            #return int(np.floor((v - bound[0]) / abs(bound[1] - bound[0]))) + 1
            for i, b in enumerate(bound):
                if v < b:
                    return i
            return l
    return [in_bound(b, v) for b, v in zip(bounds, vs)]

def discretise(bounds, vs):
    def in_bound(bound, v):
       l = len(bound)
       if l == 1 or v <= bound[0]:
            return 0
       elif v > bound[l - 1]:
            return l
       else:
            return int(np.floor((v - bound[0]) / abs(bound[1] - bound[0]))) + 1
    return [in_bound(b, v) for b, v in zip(bounds, vs)]

def plotRewards(fn='rewards.jpg', reward='Reward'):
    ## Plot Rewards
    plt.xlabel('Episodes')
    plt.ylabel(f"Average {reward}")
    plt.title(f"Average {reward} per Episode")
    plt.legend(loc='center left')
    plt.savefig(fn) 
    plt.close()  

def subPlot(rewards, text):
    l = len(rewards)
    seg = round(l / 10)
    ag = [sum(rewards[i*seg:(i+1)*seg]) / seg for i in range(10)]
    plt.plot((np.arange(len(ag)) + 1) * seg, ag, label=text)
