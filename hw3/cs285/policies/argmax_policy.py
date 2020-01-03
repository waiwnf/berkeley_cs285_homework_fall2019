import numpy as np
import tensorflow as tf


class ArgMaxPolicy:
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[np.newaxis, ...]

        # TODO: Define what action this policy should return
        # HINT1: the critic's q_t_values indicate the goodness of observations,
        # so they should be used to decide the action to perform
        q_values = self.critic(observation)
        return tf.argmax(q_values, axis=1)