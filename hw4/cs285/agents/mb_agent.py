from collections import namedtuple

import numpy as np
import tensorflow as tf

from cs285.agents.base_agent import BaseAgent
from cs285.models.ff_model import FFModel
from cs285.policies.MPC_policy import MPCPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer


DataStats = namedtuple('DataStats', ['obs_mean', 'obs_std', 'acs_mean', 'acs_std', 'delta_mean', 'delta_std'])


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params, **kwargs):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped 

        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = [
            FFModel(agent_params['ac_dim'], agent_params['ob_dim'], agent_params['n_layers'],
                    agent_params['size'], name='dyn_model_{}'.format(i))
            for i in range(self.ensemble_size)]
        self.dyn_loss = tf.keras.losses.MeanAbsoluteError()
        self.dyn_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.agent_params['learning_rate'])
                               for _ in range(self.ensemble_size)]
        for model, optimizer in zip(self.dyn_models, self.dyn_optimizers):
            model.transition_model.compile(optimizer, self.dyn_loss)

        self.actor = MPCPolicy(self.env,
                               ac_dim=self.agent_params['ac_dim'],
                               dyn_models=self.dyn_models,
                               horizon=self.agent_params['mpc_horizon'],
                               N=self.agent_params['mpc_num_action_sequences'])

        self.replay_buffer = ReplayBuffer()
        self.data_stats = None

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data/self.ensemble_size)

        for i in range(self.ensemble_size):
            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful
            data_start_idx = i * num_data_per_ens
            data_end_idx = data_start_idx + num_data_per_ens

            observations = ob_no[data_start_idx:data_end_idx, ...]  # TODO(Q1)
            actions = ac_na[data_start_idx:data_end_idx, ...]  # TODO(Q1)
            next_observations = next_ob_no[data_start_idx:data_end_idx, ...]  # TODO(Q1)

            # use datapoints to update one of the dyn_models
            model = self.dyn_models[i]  # TODO(Q1)
            optimizer = self.dyn_optimizers[i]

            with tf.GradientTape() as tape:
                predicted_next_observations = model.get_prediction(observations, actions, self.data_stats)
                loss_tensor = self.dyn_loss(next_observations, predicted_next_observations)

            trainable_vars = model.transition_model.trainable_variables
            grads = tape.gradient(loss_tensor, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            losses.append(loss_tensor.numpy().item())

        avg_loss = np.mean(losses)
        return avg_loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_stats = DataStats(
            obs_mean=np.mean(self.replay_buffer.obs, axis=0),
            obs_std=np.std(self.replay_buffer.obs, axis=0),
            acs_mean=np.mean(self.replay_buffer.acs, axis=0),
            acs_std=np.std(self.replay_buffer.acs, axis=0),
            delta_mean=np.mean(self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            delta_std=np.std(self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0))

    def sample(self, batch_size):
        # NOTE: The size of the batch returned here is sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(batch_size*self.ensemble_size)