import gym
import numpy as np
from random import random
import utils

class particle:
    pass

class PSO:
    def __init__(self, task, choice, swarmSize=20, bestfit=-1, useCnt=True, obsNo=0, threshold=200):
        self.env = gym.make(task)
        self.dimensions = len(self.env.observation_space.low)
        self.actions = self.env.action_space.n
        self.swarmSize = swarmSize
        self.initialWeight = 1.0
        self.alpha = 0.9
        self.phiLocal = 2
        self.phiGlobal = 2
        self.maxIter = 100
        self.useCnt = useCnt
        self.obsNo = obsNo
        self.threshold = threshold
        self.particles = []
        self.best = particle()
        self.best.fit = bestfit
        self.best.pos = []
        self.choice = choice
        self.episodes = 0
        self.steps = 0
        self.rewards = []
        
    def fit(self, p, its=20):
        fitnessValues = []
        action = 0
        score = 0.0
        for _ in range(its):
            self.episodes += 1
            obs = self.env.reset()
            done = False
            fitnessValue = obs[self.obsNo]
            cnt = 0
            while not done:
                action = self.choice(np.dot(obs, p))
                obs, reward, done, _ = self.env.step(action) ## fix continuous
                self.steps += 1
                if self.useCnt:
                    cnt += 1
                    fitnessValue = cnt
                elif obs[self.obsNo] > fitnessValue:
                    fitnessValue = obs[self.obsNo]
                if done:
                    self.rewards.append(score)
                else:
                    score += reward
            fitnessValues.append(fitnessValue)
        return np.average(fitnessValues)

    def run(self):
        stats = utils.Stats()

        w = self.initialWeight
        p = particle()
        for _ in range(self.swarmSize):
            p.position = np.random.uniform(0, 1, self.dimensions)
            p.velocity = np.random.uniform(0, 1, self.dimensions)
            p.fitness = self.fit(p.position)
            p.bestpos = p.position
            p.bestfit = p.fitness
            if p.bestfit > self.best.fit:
                self.best.fit = p.bestfit
                self.best.pos = p.bestpos.copy()
            self.particles.append(p)

        for i in range(self.maxIter):
            for p in self.particles:
                p.velocity = w * p.velocity + self.phiLocal * random() * (p.bestpos - p.position) + self.phiGlobal * random() * (self.best.pos - p.position)
                x = p.position + p.velocity
                p.position = np.array([d if d >= 0.0 and d <= 1.0 else random() for d in x])
                p.fitness = self.fit(p.position)
                if p.fitness > p.bestfit:
                    p.bestpos = p.position
                    p.bestfit = p.fitness
                    if p.bestfit > self.best.fit:
                        self.best.fit = p.bestfit
                        self.best.pos = p.bestpos
            w *= self.alpha
            print(i, self.best.fit, self.best.pos)
            if self.best.fit >= self.threshold: # success condition
                break

        stats.end()
        return self.rewards, self.episodes, self.steps

def play(agent):
    for _ in range(5):
        cnt = 0
        done = False
        obs = agent.env.reset()
        while not done:
            #agent.env.render()
            cnt += 1
            action = agent.actions - 1 if np.dot(obs, agent.best.pos) > 0 else 0
            obs, reward, done, _ = agent.env.step(action)
        print('Game lasted:', cnt, 'moves')

    agent.env.close()

agent = PSO('CartPole-v0', swarmSize=5, choice=lambda v: 1 if v > 0 else 0)
t = 'CartPole'
rewards, episodes, steps = agent.run()
utils.subPlot(rewards, t)
print(f"{t} Episodes/Steps: {episodes}/{steps}")
play(agent)
#agent = PSO('CartPole-v1', swarmSize=5, threshold=500, choice=choose_cp)
agent = PSO('MountainCar-v0', swarmSize=30, bestfit=-1, useCnt=False, threshold=0.5, choice=lambda v: 2 if v > 0 else 0)
t = 'MountainCar'
rewards, episodes, steps = agent.run()
utils.subPlot(rewards, t)
print(f"{t} Episodes/Steps: {episodes}/{steps}")
play(agent)
#agent = PSO('MountainCarContinuous-v0', swarmSize=50, bestfit=-100, useCnt=False, threshold=0.5, choice=lambda v: 2 if v > 0 else 0)

utils.plotRewards('PSO')
