import numpy as np
import gym
import matplotlib.pyplot as plt
import timeit
from tile_coding import tiles, IHT

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
iht = IHT(64 * 64)

def discetise_state(state):
    return tuple(tiles(iht, 6, state))

def sample(d):
    # softmax(d)
    mx = np.max(d)
    numerator = np.exp(d - mx)
    dist = numerator / np.sum(numerator)

    # biased random choice
    rng = np.random.random()
    pos = 0
    ci = 0.0

    for i in dist:
        if rng < ci:
            break
        pos += 1
        ci += i

    return pos - 1

# Fix: numpy argmax chooses lowest index when duplicates
def argmax(d):
    mx = max(d)
    if d.count(mx) > 1:
        return np.random.choice([i for i in range(len(d)) if d[i] == mx])
    else:
        return d.index(mx)

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes, discrete_states=np.array([25, 25]), debug=True):
    # Determine size of discretized state space
    if debug:
        print("Observation space: ", env.observation_space)
        print("Range of values:", env.observation_space.low, env.observation_space.high)
    no_actions = env.action_space.n
    if debug:
        print("Action space size: ", no_actions)
    
    # Initialize Q table - Array of actions as dictionaries as less space
    Q = [{} for _ in range(no_actions)]
    
    # Initialize variables to track rewards
    total_reward_list = []
    avg_total_reward_list = []
    
    # Calculate epsilon decay
    decay = (epsilon - min_eps) / episodes

    # Some counts
    tests = 5
    success_count = 0
    reward_sum = 0.0
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialise parameters
        done = False
        total_reward = 0.0
        reward = 0.0
        action = 0 # Initialise action - there is always a 0
        state = env.reset()
        
        # Discretize old state
        old_ds = discetise_state(state)

        # Render on last few iterations
        render = i >= (episodes - tests)
    
        while not done:   
            # Remember old values
            last_state = state
            last_action = action

            # Render environment for last 'test' episodes
            if render:
                env.render()
                total_reward_list = []
                
            # What is Next Action?
            if np.random.random() < epsilon:
                #pass
                action = np.random.randint(0, env.action_space.n)
            else:
                vals = [Q[a].get(tuple(old_ds), 0.0) for a in range(no_actions)]
                #action = np.argmax(vals)
                action = argmax(vals)
                #action = sample([Q[a].get(indx, 0.0) for a in range(no_actions)])
                #print(action, vals)
                
            state, reward, done, info = env.step(action) 

            modified_reward = reward
            #modified_reward = reward + 10 * (abs(state[1]) - abs(last_state[1]))
            #modified_reward = abs(state[1]) - abs(last_state[1])
            
            # Find discrete states
            new_ds = discetise_state(state)
            
            #Allow for terminal states
            if done and state[0] >= 0.5:
                success = True
                if render :
                    success_count += 1
                Q[action][tuple(old_ds)] = modified_reward
            # Adjust Q value for current state
            else:
                success = False
                old_indx = tuple(old_ds)
                old_qv = Q[action].get(old_indx, 0.0)
                new_qv = np.max([Q[a].get(tuple(new_ds), 0.0) for a in range(no_actions)])
                delta = learning * (modified_reward + discount * new_qv - old_qv)

                try:
                    Q[action][old_indx] += delta
                except:
                    #Q[action][old_indx] = np.random.uniform() + delta
                    Q[action][old_indx] = delta
                                     
            # Update variables
            total_reward += reward
            old_ds = new_ds
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= decay
        
        # Track rewards
        total_reward_list.append(total_reward)
        
        if render:
            reward = np.sum(total_reward_list)
            reward_sum += reward
        if (i + 1) % 1000 == 0:    
            avg_reward = np.mean(total_reward_list)
            avg_total_reward_list.append(avg_reward)
            total_reward_list = []
            if debug:
                print('Episode {} Average Reward: {}'.format(i+1, avg_reward))
            
    success_ratio = success_count / tests
    avg_steps = reward_sum / tests
    space_size = np.prod(discrete_states)
    memory = np.sum([len(m) / space_size for m in Q]) / len(Q)
    env.close()
    
    return avg_total_reward_list, success_ratio, avg_steps, memory

# Run Q-learning algorithm
start_time = timeit.default_timer()
rewards_list, success_ratio, avg_steps, memory = QLearning(env, 0.2, 0.9, 0.9, 0, 5000)
elapsed = timeit.default_timer() - start_time
print('Success: {}% average: {}'.format(success_ratio * 100.0, avg_steps))

print("Count:", iht.count())

# Plot Rewards
plt.plot((np.arange(len(rewards_list)) + 1) * 100.0, rewards_list)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode')
plt.savefig('rewards.jpg') 
plt.close()  
