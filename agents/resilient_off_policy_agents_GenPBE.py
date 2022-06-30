import numpy as np
import tensorflow as tf
from tensorflow import keras

class resilient_agent():
    '''
    RESILIENT PROJECTION-BASED CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the deep resilient projection-based consensus actor-critic (RPBCAC) algorithm from Figura et al. (2021).
    The algorithm is a realization of temporal difference learning with one-step lookahead. It is an instance of decentralized learning,
    where each agent receives its own reward and observes the global state and action. The RPBCAC agent seeks to maximize a team-average
    objective function of the cooperative agents in the presence of adversaries. The RPBCAC agent employs neural networks to approximate
    the actor, critic, and team-average reward function.

    The updates are divided into four parts.

    1) The agent performs a batch local stochastic update of the critic and team-average reward through methods critic_update_local() and TR_update_local().
    2) The agent estimates the neighbors' estimation errors via projection and applies the resilient projection-based consensus update.
    3) The agent performs a stochastic update using the mean estimation errors. This is executed by methods critic_update_team() and TR_update_team().

    The code is applicable for both online and batch training. The RCAC agent further includes method get_action() to sample actions from the policy approximated by the actor network.

    ARGUMENTS: NN models for actor, critic, and team_reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
               max number of adversaries among the agent's neighbors
    '''
    def __init__(self, actor, critic, bellman, behavior_policy, team_reward,slow_lr,fast_lr,gamma=0.95, H=0):
        self.actor = actor
        self.critic = critic
        self.bellman = bellman
        self.TR = team_reward
        self.gamma = gamma
        self.H = H
        self.n_actions = self.actor.output_shape[1]
        self.fast_lr = fast_lr
        self.optimizer_fast = keras.optimizers.SGD(learning_rate=fast_lr,clipnorm=1.0,clipvalue=0.5)
        self.optimizer_bellman = keras.optimizers.SGD(learning_rate=2*fast_lr,clipnorm=1.0,clipvalue=0.5)
        self.optimizer_slow = keras.optimizers.SGD(learning_rate=slow_lr,clipnorm=1.0,clipvalue=0.5)
        self.mse = keras.losses.MeanSquaredError()
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr,clipnorm=1.0,clipvalue=0.5),loss=keras.losses.SparseCategoricalCrossentropy())
        # print(self.critic.inputs,self.critic.layers[-2].outputs)
        self.critic_features = keras.Sequential([
                                    *self.critic.layers[:-1]
                                  ])
        # keras.Model(self.critic.inputs,self.critic.layers[-2].output)
        self.TR_features = keras.Sequential([
                                    *self.TR.layers[:-1],
                                  ])
        self.critic_features.build(input_shape=self.critic.input_shape)
        self.TR_features.build(input_shape=self.TR.input_shape)

        # keras.Model(self.TR.inputs,self.TR.layers[-2].output)
        
        
        self.behavior_policy = behavior_policy

    def _resilient_aggregation(self,values_innodes):
        '''
        Sorts a vector by value, eliminates H values strictly larger or smaller than the agent's value, and computes an average of the remaining values
        Arguments: 2D np array with estimated estimation errors (size = n_agents x n_observations)
        Returns: aggregated value for each observation
        '''
        n_neighbors = values_innodes.shape[0]
        own_val = values_innodes[0]                  #get own value
        sorted_vals = tf.sort(values_innodes,axis=0)        #sort neighbors' values
        H_small = sorted_vals[self.H]
        H_large = sorted_vals[n_neighbors - self.H - 1]
        lower_bound = tf.math.minimum(H_small,own_val)
        upper_bound = tf.math.maximum(H_large,own_val)
        clipped_vals = tf.clip_by_value(sorted_vals,lower_bound,upper_bound)
        aggregated_values = tf.reduce_mean(clipped_vals,axis=0)
        
        return aggregated_values

    def critic_update_team(self,s,critic_agg):
        '''
        Stochastic update of the critic using the estimated average TD error of the neighbors
        ARGUMENTS: visited consecutive states, aggregated neighbors' TD errors
        RETURNS: training loss
        '''
        phi = self.critic_features(s)
        phi_norm = tf.math.reduce_sum(tf.math.square(phi),axis=1) + 1
        weights = 1 / (2 * self.fast_lr * phi_norm)
        self.critic_features.trainable = False
        self.critic.compile(optimizer=self.optimizer_fast,loss=self.mse)
        self.critic.train_on_batch(s,critic_agg,sample_weight=weights)

    def TR_update_team(self,sa,TR_agg):
        '''
        Stochastic update of the team-average reward function using the estimated average estimation error of the neighbors
        ARGUMENTS: visited states, team actions, agregated neighbors' estimation errors
        RETURNS: training loss
        '''
        f = self.TR_features(sa)
        f_norm = tf.math.reduce_sum(tf.math.square(f),axis=1).numpy() + 1
        weights = 1 / (2 * self.fast_lr * f_norm)
        self.TR_features.trainable = False
        self.TR.compile(optimizer=self.optimizer_fast,loss=self.mse)
        self.TR.train_on_batch(sa,TR_agg,sample_weight=weights)

    # def bellman_update_team(self,s,bellman_agg):
    #     '''
    #     Stochastic update of the critic using the estimated average TD error of the neighbors
    #     ARGUMENTS: visited consecutive states, aggregated neighbors' TD errors
    #     RETURNS: training loss
    #     '''
    #     phi = self.bellman_features(s)
    #     phi_norm = tf.math.reduce_sum(tf.math.square(phi),axis=1) + 1
    #     weights = 1 / (2 * self.fast_lr * phi_norm)
    #     self.critic_features.trainable = False
    #     self.critic.compile(optimizer=self.optimizer_fast,loss=self.mse)
    #     self.critic.train_on_batch(s,critic_agg,sample_weight=weights)


    def actor_update(self,s,ns,sa,a_local,rho_local,pretrain=False):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - estimates team-average TD errors with a one-step lookahead
        - applies the estimated team-average TD errors as sample weights to the cross-entropy gradient
        ARGUMENTS: observations, new observations, state-actions pairs, local actions
        RETURNS: training loss
        '''

        with tf.GradientTape() as tape1:
            r_team = self.TR(sa)
            V = self.critic(s)
            nV = self.critic(ns)
            global_TD_error = (r_team + self.gamma * nV - V).numpy()
            action_prob = self.actor(s)
            a_local_arr = a_local.numpy().flatten()
            # print("\n\nHaaaaaaaaaaaaaaaaaaa-------------------------------{}{}------------\n\n\n".format(action_prob.shape,a_local[0]))
            actor_loss = tf.math.reduce_mean(-tf.stop_gradient(rho_local*global_TD_error)*tf.math.log(tf.stack([action_prob[:,int(a_local_each)] for a_local_each in a_local_arr])))
            # print("\n\nHaaaaaaaaaaaaaaaaaaa-------------------------------{},{},{},{}------------\n\n\n".format(rho_local,global_TD_error,actor_loss,tf.math.log(tf.stack([action_prob[:,int(a_local_each[0].numpy())] for a_local_each in a_local]))))

            grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
            self.optimizer_slow.apply_gradients(zip(grads1, self.actor.trainable_variables))
            # self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return actor_loss
        # training_loss = self.actor.train_on_batch(s,a_local,sample_weight=global_TD_error)

        # return training_loss

    def critic_update_local(self,s,ns,r_local):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        - resets the internal critic parameters to the value prior to the stochastic update
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters
        '''
        critic_weights_temp = self.critic.get_weights()
        bellman_weights_temp = self.bellman.layers[-1].get_weights()
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            nV = self.critic(ns)
            V = self.critic(s)
            H = self.bellman(s)
            local_TD_error = (r_local + self.gamma * nV) -  V

            bellman_loss = tf.math.reduce_mean((r_local + self.gamma * nV)*H - V*tf.stop_gradient(local_TD_error) - tf.math.square(H))
            # print((tf.math.reduce_mean(-(r_local + self.gamma * nV)*H + V*H + tf.math.square(H))).shape)
            # print(((self.bellman.layers[-1].weights[0])))
            bellman_loss2 = tf.math.reduce_mean(-(r_local + self.gamma * nV)*H + V*H + tf.math.square(H)) + 0.01*tf.norm(self.bellman.layers[-1].weights[0])
            self.critic_features.trainable = True
            # self.critic.compile(optimizer=self.optimizer_fast,loss=self.mse)
            grads1 = tape1.gradient(bellman_loss, self.critic.trainable_variables)
            grads2 = tape2.gradient(bellman_loss2, self.bellman.layers[-1].trainable_variables)

            self.optimizer_fast.apply_gradients(zip(grads1, self.critic.trainable_variables))
            self.optimizer_bellman.apply_gradients(zip(grads2, self.bellman.layers[-1].trainable_variables))

            # training_hist = self.critic.fit(s,local_TD_target,batch_size=s.shape[0],epochs=5,verbose=0)
            critic_weights = self.critic.get_weights()
            bellman_weights = self.bellman.layers[-1].get_weights()
            self.critic.set_weights(critic_weights_temp)
            self.bellman.layers[-1].set_weights(bellman_weights_temp)
            # print(bellman_loss2.shape,H.shape,V.shape,s.shape)

        return critic_weights, bellman_weights, -bellman_loss2

    def TR_update_local(self,sa,r_local):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        - resets the internal team-average reward parameters to the prior value
        ARGUMENTS: state-action pairs, local rewards
        RETURNS: updated team reward parameters
        '''
        TR_weights_temp = self.TR.get_weights()
        self.TR_features.trainable = True
        self.TR.compile(optimizer=self.optimizer_fast,loss=self.mse)
        training_hist = self.TR.fit(sa,r_local,batch_size=sa.shape[0],epochs=5,verbose=0)
        TR_weights = self.TR.get_weights()
        self.TR.set_weights(TR_weights_temp)

        return TR_weights, training_hist.history['loss'][0]

    def resilient_consensus_critic_hidden(self,critic_weights_innodes):
        '''
        Resilient consensus update over the critic parameters in hidden layers
        - for each parameter, the agent clips H values larger and smaller than the agent's parameter value
        - computes a simple average of the clipped parameter values
        ARGUMENTS: list of critic parameters received from neighbors (the agent's parameters always appear in the first index)
        '''
        weights_agg = []
        for layer in zip(*critic_weights_innodes):
            weights = tf.convert_to_tensor(layer)
            weights_agg.append(self._resilient_aggregation(weights).numpy())
            # print(self.critic_features.weights,self.critic.weights)
        self.critic_features.set_weights(weights_agg[:-2])

    def resilient_consensus_TR_hidden(self,TR_weights_innodes):
        '''
        Resilient consensus update over the team-average reward parameters in hidden layers
        - for each parameter, the agent clips H values larger and smaller than the agent's parameter value
        - computes a simple average of the clipped parameter values

        ARGUMENTS: list of TR parameters received from neighbors (the agent's parameters must appear first in the list followed by its neighbors)
        '''
        weights_agg = []
        for layer in zip(*TR_weights_innodes):
            weights = tf.convert_to_tensor(layer)
            weights_agg.append(self._resilient_aggregation(weights).numpy())
        self.TR_features.set_weights(weights_agg[:-2])

    def resilient_consensus_critic(self,s,critic_weights_innodes):
        '''
        Resilient consensus update over the critic estimates
        - part of the projection-based updates
        - evaluates critic of each neighbor
        - performs resilient consensus for each critic
        ARGUMENTS: states, list of critic parameters received from neighbors (the agent's parameters always appear in the first index)
        RETURNS: aggregated critic estimate
        '''
        critic_weights_temp = self.critic.layers[-1].get_weights()
        critics = []
        for weights in critic_weights_innodes:
            self.critic.layers[-1].set_weights(weights[-2:])
            critics.append(self.critic(s))
        critics = tf.convert_to_tensor(critics)
        critic_agg = self._resilient_aggregation(critics)
        self.critic.layers[-1].set_weights(critic_weights_temp)

        return critic_agg

    def resilient_consensus_bellman(self,s,bellman_weights_innodes):
        '''
        Resilient consensus update over the critic estimates
        - part of the projection-based updates
        - evaluates critic of each neighbor
        - performs resilient consensus for each critic
        ARGUMENTS: states, list of critic parameters received from neighbors (the agent's parameters always appear in the first index)
        RETURNS: aggregated critic estimate
        '''
        bellman_weights_temp = self.bellman.layers[-1].get_weights()
        bellmans = []
        for weights in bellman_weights_innodes:
            self.bellman.layers[-1].set_weights(weights[-2:])
            bellmans.append(self.bellman(s))
        bellmans = tf.convert_to_tensor(bellmans)
        bellman_agg = self._resilient_aggregation(bellmans)
        self.bellman.layers[-1].set_weights(bellman_weights_temp)

        return bellman_agg

    def resilient_consensus_TR(self,sa,TR_weights_innodes):
        '''
        Resilient consensus update over the team-average reward estimates
        - part of the projection-based updates
        - evaluates team_average reward of each neighbor
        - performs resilient consensus for each team_average reward
        ARGUMENTS: states, list of team-average reward function parameters received from neighbors (the agent's parameters always appear in the first index)
        RETURNS: aggregated team-average reward estimate
        '''
        TR_weights_temp = self.TR.layers[-1].get_weights()
        TRs = []
        for weights in TR_weights_innodes:
            self.TR.layers[-1].set_weights(weights[-2:])
            TRs.append(self.TR(sa))
        TRs = tf.convert_to_tensor(TRs)
        TR_agg = self._resilient_aggregation(TRs)
        self.TR.layers[-1].set_weights(TR_weights_temp)

        return TR_agg

    def get_action(self,state,mu=0.0):
        '''Choose an action at the current state according to behaviour policy
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        # action_prob = self.actor.predict(state).ravel()
        action_prob = self.behavior_policy(state).numpy().ravel()
        # print(self.n_actions,action_prob)
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights(), self.bellman.weights]
