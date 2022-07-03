import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import pandas as pd
import copy

#----Memory Issues-------------------------------------------
import gc
import tracemalloc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback



class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

#------------------------------------------------------------



tf.get_logger().setLevel('ERROR')

'''
This file contains a function for training consensus AC agents in gym environments. It is designed for batch updates.
'''

def test_MARL(env,agents,args,summary_writer,exp_buffer=None):
    '''
    FUNCTION test_MARL() - testing a mixed cooperative and adversarial network of consensus AC agents including RPBCAC agents
    The agents apply actions sampled from the actor network and estimate online the team-average errors for the critic and team-average reward updates.
    At the end of a sequence of episodes, the agents update the actor, critic, and team-average reward parameters in batches. All participating agents
    transmit their critic and team reward parameters but only cooperative agents perform resilient consensus updates. The critic and team-average reward
    networks are used for the evaluation of the actor gradient. In addition to the critic and team reward updates, the adversarial agents separately
    update their local critic that is used in their actor updates.

    ARGUMENTS: gym environment
               list of resilient consensus AC agents
               user-defined parameters for the simulation
    '''
    # tracemalloc.start()
    # with open("memory_profiler.txt", "w") as file:
    #         file.write("\n")

    paths = []
    n_agents, n_states = env.n_agents, args['n_states']
    n_coop = args['agent_label'].count('Cooperative')
    gamma = args['gamma']
    in_nodes = args['in_nodes']
    max_ep_len, n_episodes, n_ep_fixed = args['max_ep_len'], args['n_episodes'], args['n_ep_fixed']
    n_epochs, batch_size, buffer_size = args['n_epochs'], args['batch_size'], args['buffer_size']
    metropolis = []
    for i in range(n_agents):
        temp = np.zeros(len(in_nodes[i]))
        temp_sum=0
        for j in range(len(in_nodes[i])):
            if in_nodes[i][j]!=i:
                # print(in_nodes,j,i,temp)
                temp[j] = 1/(1+max([len(in_nodes[i]),len(in_nodes[j])]))
                temp_sum += temp[j]
            else:
                ident = j
        temp[ident] = 1- temp_sum
        metropolis.append(temp)
    if exp_buffer:
        states = exp_buffer[0]
        nstates = exp_buffer[1]
        actions = exp_buffer[2]
        rewards = exp_buffer[3]
    else:
        states, nstates, actions, rewards, log_importance_samples, joint_importance_samples = [], [], [], [], [], []
    #---------------------------------------------------------------------------
    #'                                 TRAINING                                 '
    #---------------------------------------------------------------------------
    for t in range(n_episodes):

        i = t % n_ep_fixed
        j,  ep_returns = 0, 0


        #-------------------------------------------------------------------------------------------------------------------------------------------------
        #'                       ON POLICY TESTING                                                                                                        '
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        est_returns, mean_true_on_policy_returns, mean_true_on_policy_returns_adv = [], 0, 0
        action = np.zeros(n_agents)
        #-----------------------------------------------------------------------
        #'                       BEGINNING OF EPISODE                           '
        #-----------------------------------------------------------------------
        env.reset()
        state, _ = env.get_data()
        #-----------------------------------------------------------------------
        #'       Evaluate expected returns at the beginning of episode           '
        #-----------------------------------------------------------------------
        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                est_returns.append(agents[node].critic(state.reshape(1,state.shape[0],state.shape[1]))[0][0].numpy())
        #-----------------------------------------------------------------------
        #'                           Simulate episode                           '
        #-----------------------------------------------------------------------
        while j < max_ep_len:
            for node in range(n_agents):
                  action[node] = agents[node].get_on_policy_action(state.reshape(1,state.shape[0],state.shape[1]))
            
            env.render()
            env.step(action)

            nstate, reward = env.get_data()
            ep_returns += reward * (gamma ** j)
            j += 1
            state = np.array(nstate)
            #------------------------------------------------------------------------
            #'                             END OF EPISODE                            '
            #------------------------------------------------------------------------
            #----------------------------------------------------------------------------
            #'                           TRAINING EPISODE SUMMARY                        '
            #----------------------------------------------------------------------------
            if j == max_ep_len:
                for node in range(n_agents):
                    if args['agent_label'][node] == 'Cooperative':
                        mean_true_on_policy_returns += ep_returns[node]/n_coop
                    else:
                        mean_true_on_policy_returns_adv += ep_returns[node]/(n_agents-n_coop)
                print('| Episode: {} | Est. returns: {} | Returns: {} '.format(t,est_returns,mean_true_on_policy_returns))
                path = {
                    "True_team_returns":mean_true_on_policy_returns,
                    "True_adv_returns":mean_true_on_policy_returns_adv,
                    "Estimated_team_returns":np.mean(est_returns)
                   }
                paths.append(path)
                with summary_writer.as_default():
                    tf.summary.scalar('mean_true_on_policy_returns_', mean_true_on_policy_returns, step=t)
                    tf.summary.scalar('mean_true_on_policy_returns_adv_', mean_true_on_policy_returns_adv, step=t)
                    for iters in range(len(est_returns)):
                        tf.summary.scalar('est_returns_' + str(iters), est_returns[iters], step=t)
        t+=1


    sim_data = pd.DataFrame.from_dict(paths)
    weights = [agent.get_parameters() for agent in agents]
    return weights,sim_data
