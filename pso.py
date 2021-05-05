import gym
import numpy as np
from random import random

class particle:
    pass

class PSO:
    def __init__(self, task, swarmSize=20, bestfit=-1, useCnt=True, obsNo=0, threshold=200):
        self.env = gym.make(task)
        self.dimensions = len(self.env.observation_space.low)
        self.actions = self.env.action_space.n
        self.swarmSize = swarmSize
        self.initialWeight = 1.0
        self.alpha = 0.9
        self.phiLocal = 2
        self.phiGlobal = 2
        self.maxIter = 500
        self.useCnt = useCnt
        self.obsNo = obsNo
        self.threshold = threshold
        self.particles = []
        self.best = particle()
        self.best.bestfit = bestfit
        self.best.bestposition = []
        
    def fit(self, p, its=20):
        fitnessValues = []
        for _ in range(its):
            obs = self.env.reset()
            done = False
            fitnessValue = obs[self.obsNo]
            cnt = 0
            while not done:
                action = self.actions - 1 if np.dot(obs, p) > 0 else 0
                obs, reward, done, _ = self.env.step(action)
                if self.useCnt:
                    cnt += 1
                    fitnessValue = cnt
                elif obs[self.obsNo] > fitnessValue:
                    fitnessValue = obs[self.obsNo]
            fitnessValues.append(fitnessValue)
        return np.average(fitnessValues)

    def pso(self):
        w = self.initialWeight
        p = particle()
        for _ in range(self.swarmSize):
            p.position = np.random.uniform(0, 1, self.dimensions)
            p.velocity = np.random.uniform(0, 1, self.dimensions)
            p.fitness = self.fit(p.position)
            p.bestposition = p.position
            p.bestfit = p.fitness
            if p.bestfit > self.best.bestfit:
                self.best.bestfit = p.bestfit
                self.best.bestposition = p.bestposition.copy()
            self.particles.append(p)

        for i in range(self.maxIter):
            for p in self.particles:
                p.velocity = w * p.velocity + self.phiLocal * random() * (p.bestposition - p.position) + self.phiGlobal * random() * (self.best.bestposition - p.position)
                x = p.position + p.velocity
                p.position = np.array([d if d >= 0.0 and d <= 1.0 else random() for d in x])
                p.fitness = self.fit(p.position)
                if p.fitness > p.bestfit:
                    p.bestposition = p.position
                    p.bestfit = p.fitness
                    if p.bestfit > self.best.bestfit:
                        self.best.bestfit = p.bestfit
                        self.best.bestposition = p.bestposition
            w = w * self.alpha
            print(i, self.best.bestfit, self.best.bestposition)
            if self.best.bestfit >= self.threshold: # success condition
                break

agent = PSO('CartPole-v0', swarmSize=5)
#agent = PSO('CartPole-v1', swarmSize=5, threshold=500)
#agent = PSO('MountainCar-v0', swarmSize=50, bestfit=-100, useCnt=False, threshold=0.5)
agent.pso()

for _ in range(5):
    cnt = 0
    done = False
    obs = agent.env.reset()
    while not done:
        agent.env.render()
        cnt += 1
        action = agent.actions - 1 if np.dot(obs, agent.best.bestposition) > 0 else 0
        obs, reward, done, _ = agent.env.step(action)
    print('Game lasted:', cnt, 'moves')

agent.env.close()
