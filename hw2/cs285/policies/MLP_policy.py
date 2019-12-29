import tensorflow as tf
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp
import tensorflow_probability as tfp


class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.logstd = self.add_weight(shape=(input_dim,), initializer='zeros', trainable=True, name='logstd')

    def call(self, inputs, **kwargs):
        return tf.exp(self.logstd) * tf.random.normal(tf.shape(inputs), 0, 1)


class BaseMLPPolicy(BasePolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__()
        self.model = build_mlp((ob_dim,), output_size=ac_dim, n_layers=n_layers, size=size, name='model')
    
    def save_layers(self, filepath):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(filepath)

    def restore_layers(self, filepath):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(filepath)

    def call(self, obs, **kwargs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = tf.expand_dims(obs, axis=0)
        
        return self.model(observation)
    
    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        return self.call(obs)


class DiscreteMLPPolicy(BaseMLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
    
    def call(self, obs, **kwargs):
        logits = super().call(obs, **kwargs)
        return tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=1)
    
    def get_log_prob(self, observations, actions):
        logits = super().call(observations)
        return tfp.distributions.Categorical(logits=logits).log_prob(actions)


class ContinuousMLPPolicy(BaseMLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.gauss_noise = GaussianNoise(ac_dim, name='noise')

    def call(self, obs, **kwargs):
        action_mean = super().call(obs, **kwargs)
        return action_mean + self.gauss_noise(action_mean)
    
    def get_log_prob(self, observations, actions):
        means = super().call(observations)
        scale_diag = tf.exp(self.gauss_noise.logstd)
        return tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=scale_diag).log_prob(actions)
