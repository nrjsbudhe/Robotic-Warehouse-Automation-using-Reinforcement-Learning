import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gym.envs.registration import register

# 1) observation_spaces - change the argument values
# 2) may need to change the action space to add the ordinal directions
# 3) modify goal position


def register_env():
    register(id="WarehouseEnv-v0", entry_point="env:WarehouseEnvironment", max_episode_steps = 1000)
    

class WarehouseEnvironment(gym.Env):
    def __init__(self, grid_size, materials, num_obstacles):
        super(WarehouseEnvironment, self).__init__() # Here we are calling itself, Basically calling the constructor of the parent class - WarehouseEnvironment and initializing the object using the constructor of the parent class
        self.grid_size = grid_size
        self.idx = 0
        self.materials = materials
        # self.num_materials = num_materials
        self.num_obstacles = num_obstacles
        self.action_space = spaces.Discrete(4) # Currently only 4 directions are allowed - up, right, down and left
        self.observation_space = spaces.MultiDiscrete([
            grid_size, grid_size,  # Robot position
            grid_size, grid_size,  # Goal position
            grid_size, grid_size,  # Material positions
        ])
        self.robot_position = (0,0)
        self.goal_position = (self.grid_size-1,self.grid_size-1)
        self.material_positions = [(6,0),(7,4),(2,9)]
        self.obstacle_positions = [(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),
                                    (2,3),(3,3),(4,3),(5,3),(6,3),(7,3),
                                    (2,6),(3,6),(4,6),(5,6),(6,6),(7,6),
                                    (2,7),(3,7),(4,7),(5,7),(6,7),(7,7)
                                   ]

    def generate_random_positions(self, num_positions):
        positions = set() # positions of materials and obstacles which are generated randomly so that we dont have to generate different environments for every new version
        while len(positions) < num_positions:
            position = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if position!=self.robot_position and position!=self.goal_position:
                positions.add(position) # So, the logic is that the random material or obstacle positions can be anything in the grid except for the robot location and also the goal locations. If not then add to the sets
        return list(positions)
    
    def reset(self):
        self.material_positions = [(6,0),(7,4),(2,9)]
        self.robot_position = (0,0)
        return self.get_state()
    
    def get_state(self):
        # unpacking the elements of robot_positin, goal_position and num_material_position iterables into a single tuple, finally converting it to array
        return np.array([self.robot_position[0], self.robot_position[1], self.goal_position[0], self.goal_position[1],*sum(self.material_positions, ())])  # Flatten the list of material positions

    def is_valid_move(self, position):
        # Checking if the current positions x and y after taking action is between 0 to largest possible integer and position also should not be colliding to obstacles 
        return 0 <= position[0] < self.grid_size and 0<=position[1]<self.grid_size and position not in self.obstacle_positions
    
    def take_action(self, action):
        new_position = self.robot_position
        if action == 0: #Go up
            new_position=(self.robot_position[0]-1, self.robot_position[1])
        elif action == 1: #Go down
            new_position=(self.robot_position[0]+1, self.robot_position[1])
        elif action == 2: #Go left
            new_position=(self.robot_position[0], self.robot_position[1]-1)
        elif action ==3: #Go right
            new_position=(self.robot_position[0], self.robot_position[1]+1)
        if self.is_valid_move(new_position):
            self.robot_position = new_position

    def is_goal_reached(self):
        return self.robot_position == self.goal_position
    
    def collect_material(self):
        if self.robot_position in self.material_positions:
            self.material_positions.remove(self.robot_position)
            return True
        return False
    
    def step(self, action):
        self.previous_position = self.robot_position
        self.take_action(action)
        if self.robot_position == self.goal_position:
            if len(self.material_positions) == 0:  
                reward = 5
            else:
                reward=0
                self.robot_position = self.previous_position
        elif self.collect_material():
            reward = 3
        else:
            reward = 0
        done = self.is_goal_reached() and len(self.material_positions) == 0
        return self.get_state(), reward, done, {} #empty dictionary in case we need to return additional information also called as info....
    
    def render(self, mode): 

        # Rendering environment into human readable form
        if mode == 'human':
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    position = (i,j)
                    if position == self.robot_position:
                        print('R', end=' ')
                    elif position == self.goal_position:
                        print('G', end=' ')
                    elif position in self.material_positions:
                        print('M', end=' ')
                    elif position in self.obstacle_positions:
                        print('O', end=' ')
                    else:
                        print('-', end=' ')
                print()
        
        # Rendering environment into matplotlib form
        elif mode == 'matplotlib':
            fig, ax = plt.subplots()
            ax.set_xlim([0, self.grid_size])
            ax.set_ylim([0, self.grid_size])  
            ax.invert_yaxis()

            for obstacle_position in self.obstacle_positions:
                obstacle_rect = Rectangle((obstacle_position[1], obstacle_position[0]), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(obstacle_rect)

            for material_position in self.material_positions:
                material_rect = Rectangle((material_position[1], material_position[0]), 1, 1, linewidth=1, edgecolor='green', facecolor='green')
                ax.add_patch(material_rect)

            goal_rect = Rectangle((self.goal_position[1], self.goal_position[0]), 1, 1, linewidth=1, edgecolor='red', facecolor='red')
            ax.add_patch(goal_rect)

            robot_rect = Rectangle((self.robot_position[1], self.robot_position[0]), 1, 1, linewidth=1, edgecolor='blue', facecolor='blue')
            ax.add_patch(robot_rect)

            # Adding grid for better visualization
            ax.set_xticks(np.arange(0,self.grid_size+1, 1))
            ax.set_yticks(np.arange(0,self.grid_size+1, 1))
            ax.grid(which='both',color='black', linestyle='-',linewidth=0.7)

            self.idx += 1
            plt.show()
            # plt.savefig(f"img_2_{self.idx}")
