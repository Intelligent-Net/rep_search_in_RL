import gym
import numpy as np
import math
import utils
import sys

class Agent():
    def __init__(self, env, buckets, n_episodes=1000, min_alpha=0.1, min_epsilon=0.1, gamma=0.9, alpha=0.2, epsilon=0.1, decay=0.05, cheat=False, sticky=False, softmax=False, shape_reward=False, lb=None, ub=None, balance=False):
#    def __init__(self, env, buckets, n_episodes=1000, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, decay=0.025, cheat=False, sticky=False, softmax=False, shape_reward=False, lb=None, ub=None, balance=False):
        self.env = gym.make(env)
        self.n_episodes = n_episodes
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.cheat = cheat
        self.sticky = sticky
        self.softmax = softmax
        self.balance = balance
        self.mc_reward_shaping = shape_reward and env == 'MountainCar-v0'
        self.episode_rewards = []
        self.episodes = 0
        self.steps = 0

        # initialising Q-table
        self.Q = np.zeros(buckets + (self.env.action_space.n,))
        print(self.Q.shape)
        lb = self.env.observation_space.low if lb == None else utils.set_range(lb, self.env.observation_space.low)
        ub = self.env.observation_space.high if ub == None else utils.set_range(ub, self.env.observation_space.high)
        self.bounds = utils.bound(buckets, lb, ub)
        print(lb, ub)
        self.buckets = buckets
        self.lb = lb
        self.ub = ub

    # Discretizing input space to make Q-table and to reduce dimmensionality
    def discretize(self, obs):
        return utils.discretise2(self.buckets, obs, self.lb, self.ub)
        #return utils.discretise(self.bounds, obs)

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new):
        if self.balance:
            lr = (1.0 - self.alpha) * self.Q[state_old][action]
        else:
            lr = self.Q[state_old][action]
        self.Q[state_old][action] = lr + self.alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_decay(self, e, mn):
        #return 5 * mn / (e + 1) if mn > 0.0 else 0.0
        return max(mn, min(1.0, 1.0 - math.log((e + 1) * self.decay)))

    def run(self):
        stats = utils.Stats()
        erewards = 0.0
        epsilon = self.epsilon
        for e in range(self.n_episodes):
            # As states are continuous, discretize them into buckets
            last_obs = self.env.reset()
            self.episodes +=1
            current_state = self.discretize(last_obs)

            # Get adaptive learning alpha and epsilon decayed over time
            alpha = self.get_decay(e, self.min_alpha)
            epsilon = self.get_decay(e, self.min_epsilon)
            #print(epsilon)
            done = False
            i = 0
            action = 0
            score = 0.0

            while not done:
                #if e >= self.n_episodes - 5:
                #    self.env.render()

                # Choose action according to greedy policy and take it
                if self.sticky:
                    action = action if np.random.random() <= epsilon else utils.argmax(self.Q[current_state])
                elif self.softmax:
                    action = utils.softmax(self.Q[current_state])
                else:
                    action = self.env.action_space.sample() if np.random.random() <= epsilon else utils.argmax(self.Q[current_state])

                obs, reward, done, _ = self.env.step(action)
                self.steps += 1
                new_state = self.discretize(obs)

                if self.mc_reward_shaping:
                    #print(reward, abs(obs[1]) - abs(last_obs[1]))
                    modified_reward = reward + (abs(obs[1]) - abs(last_obs[1]))
                    #modified_reward = (abs(obs[1]) - abs(last_obs[1]))
                    #modified_reward = reward
                else:
                    modified_reward = -200 if self.cheat and done else reward

                # Update Q-Table
                self.update_q(current_state, action, modified_reward, new_state)
                current_state = new_state
                last_obs = obs
                i += 1
                score += reward
                erewards += reward

            self.episode_rewards.append(score)
            #epsilon -= 5 * epsilon / (e + 1) if epsilon > 0 else 0

            if e >= self.n_episodes - 5:
                print("Final Episode ", e + 1, "iterations", i)
            if (e + 1) % 100 == 0:
                print("Episode ", e + 1, "average score is ", erewards / 100)
                erewards = 0

        stats.end()
        return self.episode_rewards, self.episodes, self.steps

lb = [-1, -0.5, -1, -math.radians(50)]
ub = [-1, 0.5, -1, math.radians(50)]
agent = Agent('CartPole-v0', (1,1,6,12), n_episodes=1000, balance=False, softmax=True, gamma=1.0, lb=lb, ub=ub)
agent.run()
if sys.argv[1] == 'CP':
    lb = [-1, -0.5, -1, -math.radians(50)]
    ub = [-1, 0.5, -1, math.radians(50)]
    agent = Agent('CartPole-v0', (8,8,6,12), gamma=1.0, n_episodes=2000, lb=lb, ub=ub)
    t = 'Simple'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('CartPole-v0', (8,8,6,12), gamma=1.0, cheat=True, n_episodes=2000, lb=lb, ub=ub)
    t = 'Shaping'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('CartPole-v0', (1,1,6,12), n_episodes=2000, gamma=1.0, lb=lb, ub=ub)
    t = 'Pole'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('CartPole-v0', (8,8,6,12), gamma=1.0, balance=True, n_episodes=2000, lb=lb, ub=ub)
    t = 'Simple Bal'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('CartPole-v0', (8,8,6,12), gamma=1.0, balance=True, cheat=True, n_episodes=2000, lb=lb, ub=ub)
    t = 'Shaping Bal'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('CartPole-v0', (1,1,6,12), balance=True, n_episodes=2000, gamma=1.0, lb=lb, ub=ub)
    t = 'Pole Bal'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    #agent = Agent('CartPole-v0', (1,1,15,25), n_episodes=1000, gamma=1.0, lb=lb, ub=ub)
    #agent = Agent('CartPole-v0', (8,8,6,12), softmax=True, n_episodes=5000, gamma=1.0, lb=lb, ub=ub)
    #agent = Agent('MountainCar-v0', (25,25), n_episodes=5000)
    utils.plotRewards('CartPole')
if sys.argv[1] == 'MC':
    agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, balance=False)
    t = 'Simple'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, sticky=True, balance=False)
    t = 'Momentum'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, shape_reward=True, balance=False)
    t = 'Shaped'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, balance=True)
    t = 'Simple Bal'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, sticky=True, balance=True)
    t = 'Momentum Bal'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")
    agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, shape_reward=True, balance=True)
    t = 'Shaped Bal'
    rewards, episodes, steps = agent.run()
    utils.subPlot(rewards, t)
    print(f"{t} Episodes/Steps: {episodes}/{steps}")

    utils.plotRewards('MountainCar')
    #agent = Agent('MountainCar-v0', (25,25), n_episodes=100, shape_reward=True, balance=True)
    #agent = Agent('MountainCar-v0', (25,25), n_episodes=1000, shape_reward=True, sticky=True, balance=True)
    #agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, softmax=True, balance=True)
