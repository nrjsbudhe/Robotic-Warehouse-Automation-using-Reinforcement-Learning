import numpy as np
import env
from collections import defaultdict
import matplotlib.pyplot as plt
import random


def argmax(arr) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    max_value = np.max(arr)
    max_indices = np.where(arr == max_value)[0]
    max_ = random.choice(max_indices)
    
    return max_

def monte_carlo(env, num_episodes, gamma=0.99):
    Q = defaultdict()
    N = defaultdict()

    returns = []
    lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()  # setting the robot/agent state to start
        episode_states = []
        episode_actions = []
        episode_rewards = []


        
        # Generating the episode with a policy
        done = False
        
        while not done:
            state_tuple = tuple(state)
            if state_tuple not in Q.keys():
                    Q[state_tuple] = np.zeros(env.action_space.n)
                    N[state_tuple] = np.zeros(env.action_space.n)
                    
            if np.random.rand() < 0.1:
                action = np.random.choice(env.action_space.n)
            else:
                #print(Q[state_tuple])
                action = argmax(Q[state_tuple])
            

            next_state, reward, done, _ = env.step(action)
            
            episode_states.append(state_tuple)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
            # print(f"Episode: {episode}, State: {state}, Action: {action}, Done: {done}")

        G = 0
        visited_states = set()

        lengths.append(len(episode_states))

        for t in range(len(episode_states) - 1, -1, -1):
            state_t = tuple(episode_states[t])  # Convert to tuple
            action_t = episode_actions[t]
            reward_t = episode_rewards[t]
            
            # Unless S_t appears in visited_states:
            if state_t not in visited_states:
                G = gamma * G + reward_t
                visited_states.add(state_t)
                N[state_t][action_t] += 1
                Q[state_t][action_t] += (1 / N[state_t][action_t]) * (G - Q[state_t][action_t])

        returns.append(G)

        def get_optimal_policy():
            policy = {}
            for state in Q.keys():
                policy[state] = argmax(Q[state])
            return policy

    return get_optimal_policy(), returns, lengths