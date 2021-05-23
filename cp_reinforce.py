import gym
import numpy as np
import copy 
import utils

# Hyperparameters
#EPISODES = 10000
EPISODES = 5000
alpha = 0.000025
gamma = 0.99

# Create gym and seed numpy
env = gym.make('CartPole-v0')
nA = env.action_space.n
np.random.seed(1)

# Init weight
w = np.random.rand(4, 2)

# Keep stats for final print of graph
episode_rewards = []

# Our policy that maps state to action parameterized by w
def policy(state,w):
    z = state.dot(w)
    exp = np.exp(z)
    #exp = z + np.min(z)     # This works too... and is slower to converge
    return exp/np.sum(exp)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# Main loop 
# Make sure you update your weights AFTER each episode
for e in range(EPISODES):

    state = env.reset()
    state = state[None,:]

    grads = []
    rewards = []

    # Keep track of game score to print
    score = 0
    done = False

    while not done:
        # Uncomment to see your model train in real time (slower)
        if e > EPISODES - 5:
            #env.render()
            pass

        # Sample from policy and take action in environment
        probs = policy(state,w)
        action = np.random.choice(nA,p=probs[0])
        next_state,reward,done,_ = env.step(action)
        next_state = next_state[None,:]

        # Compute gradient and save with reward in memory for our weight updates
        if False:
            dsoftmax = softmax_grad(probs)[action,:]
            dlog = dsoftmax / probs[0,action]
            grad = state.T.dot(dlog[None,:])
        else:
            grad = np.array([state[0] * ((1.0 if j == action else 0.0) - probs[0][j]) for j in range(nA)]).T

        grads.append(grad)
        rewards.append(reward)

        score += reward
        state = next_state

    # Weight update
    for i in range(len(grads)):
    # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
        w += alpha * grads[i] * sum([ r * (gamma ** r) for t,r in enumerate(rewards[i:])])
    
    # Append for logging and print
    episode_rewards.append(score) 
    if (e + 1) % 1000 == 0:
        print("Episode:", e + 1, "Score:", score)

utils.subPlot(episode_rewards, "CartPole Reinforce")
utils.plotRewards("Reinforce")
#plt.plot(np.arange(EPISODES),episode_rewards)
#plt.show()
#env.close()
