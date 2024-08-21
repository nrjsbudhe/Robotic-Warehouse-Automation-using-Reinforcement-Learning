import env
import Q_learning
import DQN
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import SARSA
import torch
import torch.nn as nn
import torch.nn.functional as F
import MC
import numpy as np

def plot_graph(returns, name):
    plt.plot(np.average(returns,axis=0),label = "Average over trials")
    plt.title(name)
    plt.xlabel("Episodes")
    plt.ylabel("Returns (G)")
    plt.legend()
    plt.show()

def visualize_environment(environment):
    environment.render(mode='human')
    environment.render(mode='matplotlib')



def train_dqn(env):

    gamma = 0.99

    # we train for many time-steps;  as usual, you can decrease this during development / debugging.
    # but make sure to restore it to 1_500_000 before submitting.
    num_steps = 1_500_000
    num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 200_000
    replay_prepopulate_steps = 50_000

    batch_size = 64
    exploration = DQN.ExponentialSchedule(1.0, 0.01, 1_000_000)

    # this should take about 90-120 minutes on a generic 4-core laptop
    dqn_models_c, returns_c, lengths_c, losses_c = DQN.train_dqn(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )

    assert len(dqn_models_c) == num_saves
    assert all(isinstance(value, DQN) for value in dqn_models_c.values())

    # saving computed models to disk, so that we can load and visualize them later.
    checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models_c.items()}
    torch.save(checkpoint, f'checkpoint_{env.spec.id}.pt')

if __name__ == "__main__":


    '''
    # DQN
    # env = env.WarehouseEnvironment(grid_size=3, num_materials=2, num_obstacles=2)
    # input_size = 8
    # agent = DQN.DQNAgent(state_size= input_size, action_size=env.action_space.n)
    # train_dqn(env=env)
    '''

    # Setup the environment
    env.register_env()
    env = gym.make('WarehouseEnv-v0',grid_size=10, materials=[], num_obstacles=0)
    visualize_environment(env)
   
    num_episodes = 10000
    num_trials = 5

    returns = np.zeros([num_trials,num_episodes])
    lengths = np.zeros([num_trials,num_episodes])

    algorithms = ["MC", "Q-Learning", "SARSA"]


    selected_algorithm = algorithms[2]
    
    if selected_algorithm == "MC":
        '''
            MONTE CARLO
        '''
        for trial in range(num_trials):
            print("Trial: ", trial)
            policy, returns[trial][:], lengths[trial][:] = MC.monte_carlo(env, num_episodes=num_episodes, gamma=0.95)

        plot_graph(returns, "Monte Carlo")

    elif selected_algorithm == "Q-Learning":
        '''
            Q-LEARNING
        '''
        for trial in range(num_trials):
            print("Trial: ", trial)
            policy, returns[trial][:], lengths = Q_learning.q_learning(env=env,num_episodes=num_episodes,gamma=0.9,epsilon=0.1,step_size=0.5)

        plot_graph(returns, "Q-Learning")

    elif selected_algorithm == "SARSA":
        '''
            SARSA
        '''
        for trial in range(num_trials):
            print("Trial: ", trial)
            policy, returns[trial][:], lengths = SARSA.sarsa(env=env,num_episodes=num_episodes,gamma=0.9,epsilon=0.1,step_size=0.5)

        plot_graph(returns, "SARSA")


    # Print and display the optimal policy 
    state = env.reset()
    state_t = tuple(state)
    pos = np.array([env.goal_position[0], env.goal_position[1], env.goal_position[0], env.goal_position[1]])
    pos_t = tuple(pos)


    while state_t != pos_t: 
        print(state_t)
        action = policy[state_t]
        next_state, reward, done, _ = env.step(action)
        state = next_state 
        state_t = tuple(state)
    
    env.render(mode="matplotlib")
    