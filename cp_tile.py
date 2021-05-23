import gym
import numpy as np
import pandas
from utils import *
from tile_coding import tiles, IHT

env = gym.make('CartPole-v0')
print("Observation space: ", env.observation_space)
print("Range of values:", env.observation_space.low, env.observation_space.high)
num_actions = env.action_space.n
print("Action space size: ", num_actions)
episodes = 1000
tests = 5

class QLearning:
    def __init__(self, actions, epsilon=0.1, gamma=0.90, alpha=0.5, minEpsilon=0.0):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration threshold
        self.actions = actions
        self.qs = {}
        self.minEpsilon = minEpsilon
        self.episode_rewards = []

        # Descretise Space
        self.iht = IHT(128 * 128)

    def getAction(self, state, action):
        q = np.array([self.qs.get(state, {a:0.0}).get(a, 0.0) for a in self.actions])

        # Softmax sample from policy works better than epsilon greedy?
        if np.random.random() < self.epsilon:
            return softmax(q)
            #return ucb(q, num_actions)
            #return np.random.randint(0, len(self.actions))
            #return action
            #pass

        return argmax(q)

    def update(self, state, action, nextState, reward):
        if state not in self.qs:
            self.qs[state] = {action:reward}
        elif action not in self.qs[state]:
            self.qs[state][action] = reward
        else:
            mx = max([self.qs.get(nextState, {a:0.0}).get(a, 0.0) for a in self.actions])
            delta = reward + self.gamma * mx - self.qs[state][action]
            self.qs[state][action] += self.alpha * delta

    def digitise(self, state):
        return tuple(tiles(self.iht, 4, state[2:]))

    def learn(self):
        score = [0] * 100
        scoreIndex = 0
        timeSteps = np.ndarray(0)
        epsilon = self.epsilon
        decay = (epsilon - self.minEpsilon) / episodes

        for episode in range(episodes):
            state = env.reset()
            done = False
            action = 0
            t = 0
            render = episode >= (episodes - tests)

            while not done:
                #if render:
                #    env.render()

                # choose an action
                stateId = self.digitise(state)
                action = self.getAction(stateId, action)

                # perform the action
                state, reward, done, info = env.step(action)
                nextStateId = self.digitise(state)

                if done:
                    if episode < 100:
                        reward = -200
                    self.update(stateId, action, nextStateId, reward)

                    score[scoreIndex] = t
                    scoreIndex += 1
                    if scoreIndex >= 100:
                        scoreIndex = 0
                    if (episode + 1) % 100 == 0:
                        print("Episode ",episode + 1, "finished after {} timesteps".format(t+1), "last 100 average:" ,sum(score)/len(score))
                    timeSteps = np.append(timeSteps, [int(t + 1)])
                    break
                else:
                    self.update(stateId, action, nextStateId, reward)
                t += 1

            self.episode_rewards.append(t)

        # Decay epsilon
        if epsilon > self.minEpsilon:
            epsilon -= decay

        return timeSteps, self.episode_rewards

agent = QLearning(actions=range(env.action_space.n), alpha=0.05, gamma=0.90, epsilon=0.1)

timeSteps, rewards = agent.learn()
subPlot(rewards, "CartPole")
plotRewards('Tile_CartPole')

print("Count:", agent.iht.count())
