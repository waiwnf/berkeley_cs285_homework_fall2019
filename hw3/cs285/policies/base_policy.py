import tensorflow as tf


class BasePolicy(tf.keras.Model):

    def get_action(self, obs):
        raise NotImplementedError()

    def get_log_prob(self, observations, actions):
        raise NotImplementedError()

    def save_layers(self, filepath):
        raise NotImplementedError()

    def restore_layers(self, filepath):
        raise NotImplementedError()
