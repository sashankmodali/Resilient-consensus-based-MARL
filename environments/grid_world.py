import numpy as np
import gym
from gym import spaces
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

class Grid_World(gym.Env):
    """
    Multi-agent grid-world: cooperative navigation 2
    This is a grid-world environment designed for the cooperative navigation problem. Each agent seeks to navigate to the desired position. The agent chooses one of five admissible actions
    (stay,left,right,down,up) and makes a transition only if the adjacent cell is not occupied. It receives a reward equal to the L1 distance between the visited cell and the target.
    ARGUMENTS:  nrow, ncol: grid world dimensions
                n_agents: number of agents
                desired_state: desired position of each agent
                initial_state: initial position of each agent
                randomize_state: True if the agents' initial position is randomized at the beginning of each episode
                scaling: determines if the states are scaled
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, nrow = 5, ncol=5, n_agents = 1, desired_state = None,initial_state = None,randomize_state = True,scaling = False):
        self.nrow = nrow
        self.ncol = ncol
        self.n_agents = n_agents
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.randomize_state = randomize_state
        self.n_states = 2
        self.actions_dict = {0:np.array([0,0]), 1:np.array([-1,0]), 2:np.array([1,0]), 3:np.array([0,-1]), 4:np.array([0,1])}
        
        self.cm = plt.get_cmap('gist_rainbow')
        
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        # im = plt.imread("bird.jpg")
        self.fig, self.ax = plt.subplots()
        
        # im = ax.imshow(im, extent=[0, 300, 0, 300])
        # ax.plot(x, x, ls='dotted', linewidth=2, color='red')
        # plt.axis('off')

        self.reset()

        if scaling:
            x,y=np.arange(nrow),np.arange(ncol)
            self.mean_state=np.array([np.mean(x),np.mean(y)])
            self.std_state=np.array([np.std(x),np.std(y)])
        else:
            self.mean_state,self.std_state=0,1

    def reset(self):
        '''Resets the environment'''
        if self.randomize_state:
            state = np.zeros((5,2))
            while len(np.unique(state,axis=0)) != len(arr):
                state = np.random.randint([self.nrow,self.ncol],size=(self.n_agents,self.n_states))
            self.state = state
        else:
            self.state = np.array(self.initial_state)
        self.reward = np.zeros(self.n_agents)

        return self.state

    def step(self, action):
        '''
        Makes a transition to a new state and evaluates all rewards
        Arguments: global action
        '''
        for node in range(self.n_agents):
            move = self.actions_dict[action[node]]
            dist_to_goal = np.sum(abs(self.state[node]-self.desired_state[node]))
            self.state[node] = np.clip(self.state[node] + move,0,self.nrow - 1)
            dist_to_agents = np.min(np.sum(abs(self.state-self.state[node]),axis=1))
            dist_to_goal_next = np.sum(abs(self.state[node]-self.desired_state[node]))

            if dist_to_agents > 0: #agent moves to a new cell
                self.reward[node] = - dist_to_goal_next
            elif dist_to_goal == 0 and action[node] == 0:
                self.reward[node] = 0
            else:
                self.reward[node] = - dist_to_goal - 1

    def get_data(self):
        '''
        Returns scaled reward and state, and flags if the agents have reached the target
        '''
        state_scaled = (self.state - self.mean_state) / self.std_state
        reward_scaled = self.reward / 5
        return state_scaled, reward_scaled
    def render(self,action=None):
        img = np.ones((300,300,4))
        # print(self.state)
        self.ax.clear()
        self.ax.set_xticks([300//self.ncol*node for node in range(self.n_agents)], minor=False)
        self.ax.set_yticks([300//self.nrow*node for node in range(self.n_agents)], minor=False)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.patch.set_edgecolor('black')  
        self.ax.patch.set_linewidth('1')

        self.ax.grid(color='k')
        self.ax.imshow(np.asarray(img))
        
        width = 300//self.ncol
        height =  300//self.nrow
        
        for node in range(self.n_agents):
            xloc = 300//self.ncol*self.desired_state[node,0]
            yloc = 300//self.nrow*self.desired_state[node,1]
            rectangle = Rectangle((xloc,yloc),width,height,facecolor = self.cm(node/(self.n_agents-1)),edgecolor='k',linewidth=1)
            self.ax.add_patch(rectangle)
            
        for node in range(self.n_agents):
            xnow = 300//self.ncol*self.state[node,0]
            ynow = 300//self.nrow*self.state[node,1]
            circle = Circle((xnow+width/2,ynow+height/2),min([width/4,height/4]),facecolor = self.cm(node/(self.n_agents-1)),edgecolor='k')
            self.ax.add_patch(circle)
            if action is not None:
                pass
                # print(self.actions_dict[action[node]])
                # self.ax.arrow(xnow, ynow, width/2*self.actions_dict[action[node]][0], height/2*self.actions_dict[action[node]][1],width=0.05)
                # self.ax.annotate("here", xy=(xnow, ynow), xytext=(0, 2), arrowprops=dict(arrowstyle="->"))

        
        self.ax.legend([Patch(color=self.cm(node/(self.n_agents-1))) for node in range(self.n_agents)],
           ['{}'.format(node) for node in range(self.n_agents)],bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
        plt.pause(0.2)
        

    def close(self):
        pass