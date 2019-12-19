import tensorflow as tf
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp


class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.logstd = self.add_weight(shape=(input_dim,), initializer='zeros', trainable=True, name='logstd')

    def call(self, inputs, **kwargs):
        return tf.exp(self.logstd) * tf.random.normal(tf.shape(inputs), 0, 1)


class MLPPolicy(BasePolicy):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,  # unused for now
                 nn_baseline=False,  # unused for now
                 **kwargs):
        super().__init__()

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size

        self.mean = build_mlp((self.ob_dim,), output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        self.gauss_noise = GaussianNoise(self.ac_dim, name='noise')

    ##################################

    def call(self, obs, **kwargs):
        action_mean = self.mean(obs)
        return action_mean + self.gauss_noise(action_mean)

    ##################################

    def save_layers(self, filepath):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(filepath)

    def restore_layers(self, filepath):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(filepath)

    ##################################

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        return self(observation)
