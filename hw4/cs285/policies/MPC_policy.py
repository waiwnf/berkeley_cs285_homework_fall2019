import numpy as np
import tensorflow as tf

from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.utils import normalize

class MPCPolicy(BasePolicy):
    def __init__(self, env, ac_dim, dyn_models, reward_models, horizon, N, **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.reward_models = reward_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim)
        return np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim)).astype(np.float32)

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)
        candidate_action_sequences = tf.convert_to_tensor(candidate_action_sequences)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []
        obs_norm = normalize(obs[np.newaxis, ...], self.data_statistics.obs_mean, self.data_statistics.obs_std)
        init_obs = tf.convert_to_tensor(np.repeat(obs_norm, self.N, axis=0))

        model_trajectory_rewards = np.zeros([len(self.dyn_models), self.N])

        for model_idx, model in enumerate(self.dyn_models):
            obs_trajectories = [init_obs]

            for step in range(self.horizon):
                prev_obs = obs_trajectories[-1]
                next_obs_normalized = model([prev_obs, candidate_action_sequences[:, step, :]])
                obs_trajectories.append(next_obs_normalized)

            obs_trajectories = obs_trajectories[:-1]  # Cut off the last observation
            obs_trajectories = tf.stack(obs_trajectories, axis=1)

            obs_trajectories_flat = tf.reshape(obs_trajectories, (self.N * self.horizon, self.ob_dim))
            actions_flat = tf.reshape(candidate_action_sequences, (self.N * self.horizon, self.ac_dim))

            predicted_rewards_flat_list = [reward_model([obs_trajectories_flat, actions_flat]) for reward_model in self.reward_models]
            predicted_rewards_flat_avg_models = tf.reduce_mean(tf.stack(predicted_rewards_flat_list, axis=0), axis=0)
            predicted_rewards_per_step = tf.reshape(predicted_rewards_flat_avg_models, (self.N, self.horizon))
            model_trajectory_rewards[model_idx, ...] = tf.reduce_sum(predicted_rewards_per_step, axis=1)

            # TODO(Q2)

            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble

            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward (See files in envs to see how to call this)

        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens,N] --> N
        predicted_rewards = np.mean(model_trajectory_rewards, axis=0) # TODO(Q2)

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards) #TODO(Q2)
        best_action_sequence = candidate_action_sequences[best_index, ...] #TODO(Q2)
        action_to_take = best_action_sequence[0, ...] # TODO(Q2)
        return action_to_take[np.newaxis, ...] # the None is for matching expected dimensions
