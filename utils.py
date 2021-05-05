import numpy as np
import matplotlib.pyplot as plt

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
    x = x / tau
    # Subtract max for numeric stability
    n = np.exp(x - np.max(x))
    # Take care of both 1 and 2 dimensional cases
    #dist = n / np.sum(n) if n.ndim == 1 else n / np.sum(n, 1, keepdims=True)
    dist = n / np.sum(n, n.ndim - 1, keepdims=True)

    # biased random choice
    return np.random.choice(np.arange(len(dist)), size=1, p=dist)[0]

def plotRewards(rewards, fn='rewards.jpg'):
    # Plot Rewards
    l = len(rewards)
    seg = round(l / 10)
    ag = [sum(rewards[i*seg:(i+1)*seg]) / seg for i in range(10)]
    plt.plot((np.arange(len(ag)) + 1) * seg, ag)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.savefig(fn) 
    plt.close()  
