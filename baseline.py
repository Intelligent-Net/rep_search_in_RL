import gym
import numpy as np
import math
import utils

episode_rewards = []

class Agent():
    def __init__(self, env, buckets, n_episodes=1000, min_alpha=0.1, min_epsilon=0.0, gamma=0.9, decay=0.025, cheat=False, sticky=False, softmax=False):
        self.buckets = buckets
        self.n_episodes = n_episodes
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.decay = decay
        self.cheat = cheat
        self.sticky = sticky
        self.softmax = softmax

        self.env = gym.make(env)

        # initialising Q-table
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))
        self.lb = [-100 if buckets[i] <= 1 else b if b > -1 else -1 for i, b in enumerate(self.env.observation_space.low)]
        self.ub = [100 if buckets[i] <= 1 else b if b < 1 else 1 for i, b in enumerate(self.env.observation_space.high)]
        print(self.env.observation_space.low, self.env.observation_space.high)
        print(self.lb, self.ub)

    # Discretizing input space to make Q-table and to reduce dimmensionality
    def discretize(self, obs):
        ratios = [(ob + abs(self.lb[i])) / (self.ub[i] - self.lb[i]) for i, ob in enumerate(obs)]
        nobs = [int(round((self.buckets[i] - 1) * r)) for i, r in enumerate(ratios)]
        return tuple([min(self.buckets[i] - 1, max(0, nob)) for i, nob in enumerate(nobs)])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_decay(self, e, mn):
        return max(mn, min(1.0, 1.0 - math.log((e + 1) * self.decay)))

    def run(self):
        erewards = 0.0
        for e in range(self.n_episodes):
            # As states are continuous, discretize them into buckets
            current_state = self.discretize(self.env.reset())

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
                new_state = self.discretize(obs)

                total_reward = -200 if self.cheat and done else reward
                # Update Q-Table
                self.update_q(current_state, action, total_reward, new_state, alpha)
                current_state = new_state
                i += 1
                score += reward

                erewards += reward

            episode_rewards.append(score)

            if e >= self.n_episodes - 5:
                print("Final Episode ", e + 1, "iterations", i)
            if (e + 1) % 100 == 0:
                print("Episode ", e + 1, "average score is ", erewards / 100)
                erewards = 0

        return episode_rewards

#agent = Agent('CartPole-v0', (8,8,6,12), n_episodes=5000)
#agent = Agent('CartPole-v0', (8,8,6,12), cheat=True, n_episodes=5000)
agent = Agent('CartPole-v0', (1,1,6,12))
#agent = Agent('CartPole-v0', (8,8,6,12), softmax=True, n_episodes=5000)
#agent = Agent('MountainCar-v0', (25,25), n_episodes=5000)
#agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, sticky=True)
#agent = Agent('MountainCar-v0', (25,25), n_episodes=5000, softmax=True)

scores = agent.run()

utils.plotRewards(scores)
