import os
import datetime
import numpy as np
import gym
import argparse
import pickle
from gym import spaces
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from environments.grid_world import Grid_World
from agents.resilient_agents import resilient_agent as on_policy_resilient_agent
from agents.resilient_off_policy_agents_GenPBE import resilient_agent as off_policy_resilient_agent_GenPBE
from agents.resilient_off_policy_agents_saddle_point import resilient_agent as off_policy_resilient_agent_saddle_point
from agents.adversarial_agents import Faulty_agent, Greedy_agent, Malicious_agent
import training.train_agents_off as training
import ast
from keras.utils import plot_model

'''
Cooperative navigation problem with resilient consensus and adversarial actor-critic agents
- This is a main file, where the user selects learning hyperparameters, environment parameters,
  and neural network architecture for the actor, critic, and team reward estimates.
- The script triggers a training process whose results are passed to folder Simulation_results.
'''


def main():

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training consensus AC agents')
    parser.add_argument('--n_agents',help='total number of agents',type=int,default=5)
    parser.add_argument('--agent_label', help='classification of each agent (Cooperative,Malicious,Faulty,Greedy)',type=str, default="['Cooperative','Cooperative','Cooperative','Cooperative','Cooperative']")
    parser.add_argument('--in_nodes',help='specify a list of neighbors that transmit values to each agent (include the index of the agent as the first element)',type=str,default="[[0,1,2,3],[1,2,3,4],[2,3,4,0],[3,4,0,1],[4,0,1,2]]")
    parser.add_argument('--n_actions',help='size of action space of each agent',type=int,default=5)
    parser.add_argument('--n_states',help='state dimension of each agent',type=int,default=2)
    parser.add_argument('--n_episodes', help='Total number of episodes', type=int, default=7000)
    parser.add_argument('--max_ep_len', help='Number of steps per episode', type=int, default=20)
    parser.add_argument('--n_ep_fixed',help='Number of episodes under a fixed policy',type=int,default=50)
    parser.add_argument('--n_epochs',help='Number of updates in the policy evaluation',type=int,default=10)
    parser.add_argument('--slow_lr', help='actor network learning rate',type=float, default=0.01)
    parser.add_argument('--fast_lr', help='critic network learning rate',type=float, default=0.01)
    parser.add_argument('--batch_size', help='batch size for policy evaluation',type=int,default=200)
    parser.add_argument('--buffer_size',help='size of experience replay buffer',type=int,default=2000)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--H', help='max number of adversaries in the local neighborhood', type=int, default=0)
    parser.add_argument('--common_reward',help='Set to True if the agents receive the team-average reward',default=False)
    parser.add_argument('--summary_dir',help='Create a directory to save simulation results', default='./simulation_results/')
    parser.add_argument('--pretrained_agents',help='Set to True if the agents have been pretrained',default=False)
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=300)
    parser.add_argument('--algo',help='Choose between RPBCAC or GenPBE',type=str,default="RPBCAC")

    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])
    s_desired = np.random.randint(0,5,size=(args['n_agents'],args['n_states']))
    s_initial = np.random.randint(0,5,size=(args['n_agents'],args['n_states']))

    #----------------------------------------------------------------------------------------------------------------------------------------
    args['agent_label'] = ast.literal_eval("".join(args['agent_label']))
    args['in_nodes'] = ast.literal_eval("".join(args['in_nodes']))
    assert(len(args['in_nodes']) == args['n_agents'])
    if args['pretrained_agents']:
        pretrained_weights = np.load('pretrained_weights.npy', allow_pickle=True)
        s_desired = np.load('desired_state.npy', allow_pickle=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args['summary_dir']+'/logs/gradient_tape/' + current_time + '/train'
    test_log_dir =  args['summary_dir']+'/logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    '''NEURAL NETWORK ARCHITECTURE'''
    agents = []

    def behavior_policy(state):
        return 1./args['n_actions']*tf.ones([tf.size(state)/args['n_states']/args['n_agents'],args['n_actions']],tf.float32)

    for node in range(args['n_agents']):
        actor = keras.Sequential([
                                    keras.Input(shape=(args['n_agents'],args['n_states'])),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(args['n_actions'], activation='softmax')
                                  ])
        if args['algo'] == "GenPBE" or args['algo'] == "saddle_point":
            critic_base = keras.Sequential([
                                    keras.Input(shape=(args['n_agents'],args['n_states'])), # + args['n_actions']
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                  ])
            critic = keras.Sequential([
                                    critic_base,
                                    keras.layers.Dense(1)
                                  ])

            bellman = keras.Sequential([
                                        critic_base,
                                        keras.layers.Dense(1)
                ])
            critic.build(input_shape=(None,args['n_agents'],args['n_states']))
            bellman.build(input_shape=(None,args['n_agents'],args['n_states']))
        else:
            critic = keras.Sequential([
                                        keras.Input(shape=(args['n_agents'],args['n_states'])),
                                        keras.layers.Flatten(),
                                        keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                        keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                        keras.layers.Dense(1)
                                      ])

        team_reward = keras.Sequential([
                                    keras.Input(shape=(args['n_agents'],args['n_states'] + 1)),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(1)
                                  ])
        if args['pretrained_agents']:
            actor.set_weights(pretrained_weights[node][0])
            critic.set_weights(pretrained_weights[node][1])
            team_reward.set_weights(pretrained_weights[node][2])
            if args['algo']=='GenPBE' or args['algo'] == "saddle_point":
                bellman.layers[-1].set_weights(pretrained_weights[node][3])

        if args['agent_label'][node] == 'Malicious':        #create a malicious agent
            print("This is a malicious agent")
            agents.append(Malicious_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))
            if args['pretrained_agents']:
                agents[node].critic_local_weights = pretrained_weights[node][3]

        elif args['agent_label'][node] == 'Faulty':         #create a faulty agent
            print("This is a faulty agent")
            agents.append(Faulty_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],gamma = args['gamma']))

        elif args['agent_label'][node] == 'Greedy':         #create a greedy agent
            print("This is a greedy agent")
            agents.append(Greedy_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))

        else: # args['agent_label'][node] == 'Cooperative': #create a cooperative agent
            print("This is a cooperative agent")
            if args['algo']  == "GenPBE" :
                plot_model(critic, to_file='critic_GenPBE.png',expand_nested=True,show_shapes=True,show_layer_names=True)
                plot_model(bellman, to_file='bellman_GenPBE.png',expand_nested=True,show_shapes=True,show_layer_names=True)
                plot_model(actor, to_file='actor_GenPBE.png',expand_nested=True,show_shapes=True,show_layer_names=True)
                # print("\n\n-----{},{}-------------\n\n".format(critic.layers,critic.inputs))
                # print("\n\n-----{},{}-------------\n\n".format(critic.layers[-2].output,critic.output))
                agents.append(off_policy_resilient_agent_GenPBE(actor,critic,bellman,behavior_policy,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma'],H = args['H']))
                critic.save('critic_GenPBE.h5')
                bellman.save('bellman_GenPBE.h5')
                actor.save('actor_GenPBE.h5')
            elif args['algo'] == "saddle_point":
                plot_model(critic, to_file='critic_saddle_point.png',expand_nested=True,show_shapes=True,show_layer_names=True)
                plot_model(bellman, to_file='bellman_saddle_point.png',expand_nested=True,show_shapes=True,show_layer_names=True)
                plot_model(actor, to_file='actor_saddle_point.png',expand_nested=True,show_shapes=True,show_layer_names=True)
                critic.save('critic_saddle_point.h5')
                bellman.save('bellman_saddle_point.h5')
                actor.save('actor_saddle_point.h5')
                # print("\n\n-----{},{}-------------\n\n".format(critic.layers,critic.inputs))
                # print("\n\n-----{},{}-------------\n\n".format(critic.layers[-2].output,critic.output))
                agents.append(off_policy_resilient_agent_saddle_point(actor,critic,bellman,behavior_policy,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma'],H = args['H']))
            else:
                agents.append(on_policy_resilient_agent(actor,critic,team_reward,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma'],H = args['H']))

    print(args,s_desired)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    '''TRAIN AGENTS'''
    env = Grid_World(nrow=5,
                     ncol=5,
                     n_agents=args['n_agents'],
                     desired_state=s_desired,
                     initial_state=s_initial,
                     randomize_state=True,
                     scaling=True
                     )
    agent_weights,sim_data = training.train_MARL(env,agents,args,train_summary_writer)
    #----------------------------------------------------------------------------------------------------
    sim_data.to_pickle("sim_data.pkl")
    np.save('pretrained_weights.npy', agent_weights, allow_pickle=True)
    np.save('desired_state.npy',s_desired,allow_pickle=True)


if __name__ == '__main__':

    main()