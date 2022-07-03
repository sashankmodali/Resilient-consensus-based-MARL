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

def train_MARL(env,agents,args,summary_writer,exp_buffer=None):
    '''
    FUNCTION train_MARL() - training a mixed cooperative and adversarial network of consensus AC agents including RPBCAC agents
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
        

        #----------------------------------------------------------------------------------------------------------------------------------------------
        #'                       RPBCAC or OFF-POLICY TRAINING                                                                                         '
        #----------------------------------------------------------------------------------------------------------------------------------------------
        if i!= n_ep_fixed-2 or (args['algo'] != "GenPBE" and args['algo']!="saddle_point"):
            est_returns, mean_true_returns, mean_true_returns_adv = [], 0, 0
            action, actor_loss, critic_loss, TR_loss = np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents)

            #-----------------------------------------------------------------------
            #'                       BEGINNING OF EPISODE                           '
            #-----------------------------------------------------------------------
            env.reset()
            state, _ = env.get_data()
            #-----------------------------------------------------------------------
            #'       Evaluate expected returns at the beginning of episode          '
            #-----------------------------------------------------------------------
            for node in range(n_agents):
                if args['agent_label'][node] == 'Cooperative':
                    est_returns.append(agents[node].critic(state.reshape(1,state.shape[0],state.shape[1]))[0][0].numpy())
            #-----------------------------------------------------------------------
            #'                           Simulate episode                           '
            #-----------------------------------------------------------------------
            while j < max_ep_len:
                for node in range(n_agents):
                      action[node] = agents[node].get_action(state.reshape(1,state.shape[0],state.shape[1]))
                      while agents[node].actor(state.reshape(1,state.shape[0],-1)).numpy().ravel()[int(action[node])] < 1e-5:
                          action[node] = agents[node].get_action(state.reshape(1,state.shape[0],state.shape[1]))
                env.step(action)
                nstate, reward = env.get_data()
                ep_returns += reward * (gamma ** j)
                j += 1
                #-----------------------------------------------------------------------
                # '                    Update experience replay buffers                 '
                #-----------------------------------------------------------------------
                states.append(np.array(state))
                nstates.append(np.array(nstate))
                actions.append(np.array(action).reshape(-1,1))
                rewards.append(np.array(reward).reshape(-1,1))
                if args['algo'] =="GenPBE" or args['algo'] == "saddle_point":
                    # print(agents[node].actor(state.reshape(1,state.shape[0],-1)))
                    log_importance_sample = np.array([np.log(agents[node].actor(state.reshape(1,state.shape[0],-1)).numpy().ravel()[int(action[node])]/agents[node].behavior_policy(state).numpy().ravel()[int(action[node])]) for node in range(args['n_agents'])]).reshape(-1,1)
                    log_importance_sample_copy = copy.deepcopy(log_importance_sample)
                    log_importance_sample_copy2 = copy.deepcopy(log_importance_sample)
                    log_importance_samples.append(log_importance_sample)
                    # print("\n\n\n=====================================v========================================")
                    # print(metropolis)
                    # print(log_importance_sample_copy2,log_importance_sample_copy[in_nodes[0]])
                    while np.linalg.norm(log_importance_sample_copy[0:-1] - log_importance_sample_copy[1:]) > 1e-2:
                        for iters in range(len(log_importance_sample_copy2)):
                            log_importance_sample_copy2[iters] = np.sum(metropolis[iters]*(log_importance_sample_copy[in_nodes[iters]]).flatten())
                        log_importance_sample_copy = copy.deepcopy(log_importance_sample_copy2)
                    joint_importance_samples.append(np.exp(log_importance_sample_copy))
                    # print(log_importance_sample_copy2)
                    # print("=====================================^========================================\n\n\n")
                state = np.array(nstate)
                #------------------------------------------------------------------------
                #'                             END OF EPISODE                            '
                #------------------------------------------------------------------------
                #'                            ALGORITHM UPDATES                          '
                #------------------------------------------------------------------------
                if i == n_ep_fixed-1 and j == max_ep_len:
                    print("training===========================================================V")
                    # Convert experiences to tensors
                    s = tf.convert_to_tensor(states,tf.float32)
                    ns = tf.convert_to_tensor(nstates,tf.float32)
                    r = tf.convert_to_tensor(rewards,tf.float32)
                    a = tf.convert_to_tensor(actions,tf.float32)
                    sa = tf.concat([s,a],axis=-1)
                    rho = tf.convert_to_tensor(joint_importance_samples,tf.float32)

                    # Evaluate team-average reward of cooperative agents
                    r_coop = tf.zeros([r.shape[0],r.shape[2]],tf.float32)
                    for node in (x for x in range(n_agents) if args['agent_label'][x] == 'Cooperative'):
                        r_coop += r[:,node] / n_coop

                    for n in range(n_epochs):
                        critic_weights,TR_weights,bellman_weights, = [],[],[]
                        #--------------------------------------------------------------------
                        #'             I) LOCAL CRITIC AND TEAM-AVERAGE REWARD UPDATES       '
                        #--------------------------------------------------------------------
                        for node in range(n_agents):
                            r_applied = r_coop if args['common_reward'] else r[:,node]
                            if args['agent_label'][node] == 'Cooperative':
                                x, TR_loss[node] = agents[node].TR_update_local(sa,r_applied)
                                if args['algo']=='GenPBE' or args['algo'] == "saddle_point":
                                    y, z, critic_loss[node] = agents[node].critic_update_local(s,ns,r_applied)
                                else:
                                    y, critic_loss[node] = agents[node].critic_update_local(s,ns,r_applied)
                            elif args['agent_label'][node] == 'Greedy':
                                x, TR_loss[node] = agents[node].TR_update_local(sa,r[:,node])
                                y, critic_loss[node] = agents[node].critic_update_local(s,ns,r[:,node])
                            elif args['agent_label'][node] == 'Malicious':
                                agents[node].critic_update_local(s,ns,r[:,node])
                                x, TR_loss[node] = agents[node].TR_update_compromised(sa,-r_coop)
                                y, critic_loss[node] = agents[node].critic_update_compromised(s,ns,-r_coop)
                            elif args['agent_label'][node] == 'Faulty':
                                x = agents[node].get_TR_weights()
                                y = agents[node].get_critic_weights()
                            TR_weights.append(x)
                            critic_weights.append(y)
                            if args['algo'] == "GenPBE" or args['algo'] == "saddle_point":
                                bellman_weights.append(z)

                        #--------------------------------------------------------------------
                        #'                     II) RESILIENT CONSENSUS UPDATES               '
                        #--------------------------------------------------------------------
                        for node in (x for x in range(n_agents) if args['agent_label'][x] == 'Cooperative'):
                            #----------------------------------------------------------------
                            # '               a) RECEIVE PARAMETERS FROM NEIGHBORS            '
                            #----------------------------------------------------------------
                            critic_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                            bellman_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                            TR_weights_innodes = [TR_weights[i] for i in in_nodes[node]]
                            if args['algo']== "GenPBE" or args['algo'] == "saddle_point":
                                bellman_weights_innodes = [bellman_weights[i] for i in in_nodes[node]]
                            #----------------------------------------------------------------
                            # '               b) CONSENSUS UPDATES OF HIDDEN LAYERS           '
                            #----------------------------------------------------------------
                            agents[node].resilient_consensus_critic_hidden(critic_weights_innodes)
                            agents[node].resilient_consensus_TR_hidden(TR_weights_innodes)
                            # if args['algo']== "GenPBE" or args['algo'] == "saddle_point":
                            #     agents[node].resilient_consensus_bellman_hidden(bellman_weights_innodes)
                            #----------------------------------------------------------------
                            # '               c) CONSENSUS OVER UPDATED ESTIMATES             '
                            #----------------------------------------------------------------
                            critic_agg = agents[node].resilient_consensus_critic(s,critic_weights_innodes)
                            TR_agg = agents[node].resilient_consensus_TR(sa,TR_weights_innodes)
                            if args['algo']== "GenPBE" or args['algo'] == "saddle_point":
                                bellman_agg = agents[node].resilient_consensus_bellman(s,bellman_weights_innodes)
                            #----------------------------------------------------------------
                            # '    d) STOCHASTIC UPDATES USING AGGREGATED ESTIMATION ERRORS   '
                            #----------------------------------------------------------------
                            agents[node].critic_update_team(s,critic_agg)
                            agents[node].TR_update_team(sa,TR_agg)
                            if args['algo']== "GenPBE" or args['algo'] == "saddle_point":
                                agents[node].bellman_update_team(s,bellman_agg)
                    #--------------------------------------------------------------------
                    #'                           III) ACTOR UPDATES                      '
                    #--------------------------------------------------------------------
                    for node in range(n_agents):
                        if args['agent_label'][node] == 'Cooperative':
                            if args['algo'] == "GenPBE" or args['algo'] == "saddle_point":
                                actor_loss[node] = agents[node].actor_update(s[-max_ep_len*n_ep_fixed:],ns[-max_ep_len*n_ep_fixed:],sa[-max_ep_len*n_ep_fixed:],a[-max_ep_len*n_ep_fixed:,node],rho[-max_ep_len*n_ep_fixed:,node])
                            else:
                                actor_loss[node] = agents[node].actor_update(s[-max_ep_len*n_ep_fixed:],ns[-max_ep_len*n_ep_fixed:],sa[-max_ep_len*n_ep_fixed:],a[-max_ep_len*n_ep_fixed:,node])
                        else:
                            actor_loss[node] = agents[node].actor_update(s[-max_ep_len*n_ep_fixed:],ns[-max_ep_len*n_ep_fixed:],r[-max_ep_len*n_ep_fixed:,node],a[-max_ep_len*n_ep_fixed:,node])
                    #--------------------------------------------------------------------
                    #'                   IV) EXPERIENCE REPLAY BUFFER UPDATES             '
                    #--------------------------------------------------------------------

                    if len(states) > buffer_size:
                        q = len(states) - buffer_size
                        del states[:q]
                        del nstates[:q]
                        del actions[:q]
                        del rewards[:q]
                        del log_importance_samples[:q]
                        del joint_importance_samples[:q]
                        gc.collect()

            #----------------------------------------------------------------------------
            #'                           TRAINING EPISODE SUMMARY                        '
            #----------------------------------------------------------------------------
            for node in range(n_agents):
                if args['agent_label'][node] == 'Cooperative':
                    mean_true_returns += ep_returns[node]/n_coop
                else:
                    mean_true_returns_adv += ep_returns[node]/(n_agents-n_coop)

            print('| Episode: {} | Est. returns: {} | Returns: {} | Average critic loss: {} | Average TR loss: {} | Average actor loss: {} '.format(t,est_returns,mean_true_returns,critic_loss,TR_loss,actor_loss))
            path = {
                    "True_team_returns":mean_true_returns,
                    "True_adv_returns":mean_true_returns_adv,
                    "Estimated_team_returns":np.mean(est_returns)
                   }
            paths.append(path)
            with summary_writer.as_default():
                tf.summary.scalar('mean_true_returns_', mean_true_returns, step=t)
                for iters in range(len(est_returns)):
                    tf.summary.scalar('est_returns_' + str(iters), est_returns[iters], step=t)
                    tf.summary.scalar('critic_loss_'+ str(iters), critic_loss[iters], step=t)
                    tf.summary.scalar('TR_loss_'+ str(iters), TR_loss[iters], step=t)
                    tf.summary.scalar('actor_loss_'+ str(iters), actor_loss[iters], step=t)
                    tf.summary.scalar('ep_returns_'+ str(iters), ep_returns[iters], step=t)
                    if args['algo'] == "GenPBE" or args['algo'] == "saddle_point":
                        tf.summary.scalar('rho_local_'+ str(iters), log_importance_sample_copy.flatten()[iters], step=t)
                        if i == n_ep_fixed-1 and j == max_ep_len:
                            tf.summary.scalar('critic_agg_'+ str(iters), critic_agg.numpy().flatten()[iters], step=t)
                            tf.summary.scalar('bellman_agg_'+ str(iters), bellman_agg.numpy().flatten()[iters], step=t)
            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics("lineno")
            # with open("memory_profiler.txt", "a+") as file:
            #     file.write("-------------------------------------------\n")
            #     [file.write(str(stat) + "\n") for stat in top_stats[0:3]]
            #     file.write("\n")



        #-------------------------------------------------------------------------------------------------------------------------------------------------
        #'                       ON POLICY TESTING                                                                                                        '
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        else:
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
                if i == n_ep_fixed-2 and j == max_ep_len:
                    for node in range(n_agents):
                        if args['agent_label'][node] == 'Cooperative':
                            mean_true_on_policy_returns += ep_returns[node]/n_coop
                        else:
                            mean_true_on_policy_returns_adv += ep_returns[node]/(n_agents-n_coop)
                    with summary_writer.as_default():
                        tf.summary.scalar('mean_true_on_policy_returns_', mean_true_on_policy_returns, step=t)
                        # tf.summary.scalar('mean_true__on_policy_returns_adv_', mean_true_on_policy_returns_adv_, step=t)
                        for iters in range(len(est_returns)):
                            tf.summary.scalar('est_returns_' + str(iters), est_returns[iters], step=t)


    sim_data = pd.DataFrame.from_dict(paths)
    weights = [agent.get_parameters() for agent in agents]
    return weights,sim_data
