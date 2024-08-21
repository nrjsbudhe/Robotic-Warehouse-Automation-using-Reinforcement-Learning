import random
from collections import defaultdict
import numpy as np

def argmax(arr):

    max_value = np.max(arr)
    max_indices = np.where(arr == max_value)[0]
    max_ = random.choice(max_indices)
    
    return max_

def sarsa(
    env,
    num_episodes: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """

    Q = defaultdict()

    def policy(state):
        state_tuple = tuple(state)
        if np.random.rand() < epsilon:
            return env.action_space.sample()  # Exploration
        else:
            return argmax(Q[state_tuple])
        
    done = False
    returns = []
    lengths = []
    state = env.reset()

    for episode in range(num_episodes):
        state = env.reset()
        t_episode = 0
        G = 0
        done = False

        # Initialize the first action
        state_tuple = tuple(state)
        if state_tuple not in Q.keys():
            Q[state_tuple] = np.zeros(env.action_space.n)

        action = policy(state)

        while not done:
            t_episode += 1

            # Handle null state condition
            state_tuple = tuple(state)
            if state_tuple not in Q.keys():
                Q[state_tuple] = np.zeros(env.action_space.n)

            # Take 1 step in the enviroment
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            
            # Handle null next_state environment
            next_state_tuple = tuple(next_state)
            if next_state_tuple not in Q.keys():
                Q[next_state_tuple] = np.zeros(env.action_space.n)

            # Take next action
            next_action = policy(next_state)
            
            # Update
            Q[state_tuple][action] = Q[state_tuple][action] + step_size * (reward + gamma*Q[next_state_tuple][next_action] - Q[state_tuple][action])
            
            # Update state and return
            G += (reward + gamma*np.max(Q[state_tuple][:]))
            state = next_state

        returns.append(G)
        lengths.append(t_episode)


    def get_optimal_policy():
        policy = {}
        for state in Q.keys():
            policy[state] = argmax(Q[state])
        return policy
    

    return get_optimal_policy(), returns, lengths
