import tensorflow as tf
import numpy as np
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class MLP(Model):
    def __init__(self, output_size, scope, n_layers, size, activation = tf.nn.tanh, output_activation = None):
        """
            Builds a feedforward neural network

            arguments:
                output_size: size of the output layer
                scope: variable scope of the network
                n_layers: number of hidden layers
                size: dimension of the hidden layer
                activation: activation of the hidden layers
                output_activation: activation of the ouput layers

            returns:
                output placeholder of the network (the result of a forward pass)

            Hint: use tf.layers.dense
        """
        super(MLP, self).__init__()

        self.layers_lists = []
        for _ in range(n_layers):
            dense = Dense(size, activation = activation)
            self.layers_lists.append(dense)
        if output_activation == None:
            dense = Dense(output_size)
        else:
            dense = Dense(output_size, activation = output_activation)
        self.layers_lists.append(dense)

    def call(self, inputs, training = True):
        x = inputs
        for layer in self.layers_lists:
            x = layer(x)
        return x

def pathlength(path):
    return len(path["reward"])

class Agent():
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

        self.update_op = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.model = MLP(output_size = self.ac_dim, scope=None, n_layers = self.n_layers, size = self.size,
                         activation = tf.nn.tanh, output_activation = None)

    def policy_forward_pass(self, sy_ob_no):
        """"
            arguments:
                sy_ob_no: (batch_size, self.ob_dim)
            returns:
                    sy_logits_na: (batch_size, self.ac_dim)
        """
        outputs = self.model(sy_ob_no) # raw probabilities
        if self.discrete:
            sy_logits_na = outputs # raw value, no log
            return sy_logits_na

    def sample_action(self, policy_parameters):
        """
            arguments:
                    sy_logits_na: (batch_size, self.ac_dim)
            returns:
                sy_sampled_ac:
                    if discrete: (batch_size,)
        """
        if self.discrete:
            sy_sampled_ac = tf.random.categorical(policy_parameters, policy_parameters.shape[0])
            sy_sampled_ac = tf.reshape(sy_sampled_ac, [-1]).numpy()
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                        sy_logits_na: (batch_size, self.ac_dim)
                sy_ac_na:
                    if discrete: (batch_size,)
            returns:
                sy_logprob_n: (batch_size)

        """
        if self.discrete:
            sy_ac_na = tf.dtypes.cast(sy_ac_na, tf.int32)
            sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = sy_ac_na, logits = policy_parameters)
        return sy_logprob_n

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.01)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 3----------
            #====================================================================================#
            self.policy_parameters = self.policy_forward_pass(ob[None])
            ac = self.sample_action(self.policy_parameters)
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32),
                "reward" : np.array(rewards, dtype=np.float32),
                "action" : np.array(acs, dtype=np.float32)}
        return path

    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------

            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in
            Agent.define_placeholders).

            Recall that the expression for the policy gradient PG is

                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]

            where

                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t.

            You will write code for two cases, controlled by the flag 'reward_to_go':

              Case 1: trajectory-based PG

                  (reward_to_go = False)

                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
                  entire trajectory (regardless of which time step the Q-value should be for).

                  For this case, the policy gradient estimator is

                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

                  where

                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

                  Thus, you should compute

                      Q_t = Ret(tau)

            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above.
        """
        # YOUR_CODE_HERE

        if self.reward_to_go:
            q_n = []
            for path in re_n:
                temp = 0
                for index, rewards in enumerate(path):
                    temp = temp + (self.gamma)**index * rewards
                for index, rewards in enumerate(path):
                    if index == 0:
                        q_n.append(temp)
                    else:
                        temp = temp - (self.gamma)**index * rewards
                        q_n.append(temp)
        else:
            q_n = []
            for path in re_n:
                temp = 0
                for index, rewards in enumerate(path):
                    temp = temp + ((self.gamma)**index) * rewards
                q_n.extend([temp]*len(path))

        return np.array(q_n)

    def compute_advantage(self, ob_no, q_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Computing Baselines
        #====================================================================================#
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.
            raise NotImplementedError
            b_n = None # YOUR CODE HERE
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Advantage Normalization
        #====================================================================================#
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        #re2_n = [((re - np.mean(re)) / (np.std(re) + 1e-10)) for re in re_n]
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n, re_n):
        """
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        with tf.GradientTape() as tape:
            policy_parameters = self.model(ob_no)
            loss1 = self.get_log_prob(policy_parameters, ac_na)
            loss1 = tf.reshape(loss1, (-1,1))
            adv_n = tf.reshape(adv_n , (-1,1))
            loss2 = tf.reduce_mean(tf.multiply(loss1, adv_n))
        gradients = tape.gradient(loss2, self.model.trainable_variables)
        self.update_op.apply_gradients(zip(gradients, self.model.trainable_variables))
        print(loss2.numpy())

def train_PG(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        reward_to_go,
        animate,
        logdir,
        normalize_advantages,
        nn_baseline,
        seed,
        n_layers,
        size):


    # Make the gym environment
    env = gym.make(env_name)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)


    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        agent.update_parameters(ob_no, ac_na, q_n, adv_n, re_n)

def main():
            train_PG(
                exp_name="vpg",
                env_name="CartPole-v0",
                n_iter=100,
                gamma=1.0,
                min_timesteps_per_batch=1000,
                max_path_length=None,
                learning_rate=5e-3,
                reward_to_go=False,
                animate=True,
                logdir="",
                normalize_advantages=True,
                nn_baseline=False,
                seed=1,
                n_layers=2,
                size=64)

if __name__ == "__main__":
    main()
