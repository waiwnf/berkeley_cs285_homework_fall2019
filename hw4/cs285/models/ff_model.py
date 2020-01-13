import os

import tensorflow as tf

from .base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure.tf_utils import build_mlp


class FFModel(tf.keras.Model):

    def __init__(self, ac_dim, ob_dim, n_layers, size, name='dyn_model'):
        super().__init__()

        # TODO(Q1) Use the build_mlp function to define a neural network that predicts unnormalized delta states (i.e. change in state)
        self.transition_model = build_mlp(input_shape=(ac_dim + ob_dim,), output_size=ob_dim, n_layers=n_layers, size=size, name=os.path.join(name, 'transition_model'))

    #############################

    # Predicts normalized state transitions
    def call(self, inputs, training=None, mask=None):
        obs, acs = inputs
        concatenated_input = tf.concat([obs, acs], axis=1)
        delta_pred = self.transition_model(concatenated_input)
        return obs + delta_pred

    #############################

    def get_prediction(self, obs, acs, data_stats):
        if len(obs.shape)>1:
            observations = obs
            actions = acs
        else:
            observations = obs[None]
            actions = acs [None]

        observations_norm = normalize(observations, data_stats.obs_mean, data_stats.obs_std)
        actions_norm = normalize(actions, data_stats.acs_mean, data_stats.acs_std)
        prediction_normalized = self.call([observations_norm, actions_norm])
        prediction = unnormalize(prediction_normalized, data_stats.obs_mean, data_stats.obs_std)
        return prediction


class RewardModel(tf.keras.Model):
    def __init__(self, ac_dim, ob_dim, n_layers, size, name='reward'):
        super().__init__()
        self.reward_model = build_mlp(input_shape=(ac_dim + ob_dim,), output_size=1, n_layers=n_layers, size=size, name=os.path.join(name, 'reward_model'))

    #############################

    # Predicts normalized state transitions
    def call(self, inputs, training=None, mask=None):
        obs, acs = inputs
        concatenated_input = tf.concat([obs, acs], axis=1)
        return self.reward_model(concatenated_input)
