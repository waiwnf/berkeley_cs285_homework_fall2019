import numpy as np
import tensorflow as tf
from cs285.policies.base_policy import BasePolicy
import pickle


class Loaded_Gaussian_Policy(BasePolicy):
    def __init__(self, filename, **kwargs):
        super().__init__()

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        nonlin_type = data['nonlin_type']
        self.nonlin = None
        if nonlin_type == 'lrelu':
            self.nonlin = tf.keras.layers.LeakyReLU(alpha=.01)
        elif nonlin_type == 'tanh':
            self.nonlin = tf.tanh
        else:
            raise NotImplementedError(nonlin_type)

        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        self._load_weights()


    ##################################

    def _load_weights(self):

        # Build the policy and load the weights. First, observation normalization.
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        self.obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D'].astype(np.float32)
        self.obsnorm_meansq = self.policy_params['obsnorm']['Standardizer']['meansq_1_D'].astype(np.float32)
        self.obsnorm_stdev = np.sqrt(np.maximum(0, self.obsnorm_meansq - np.square(self.obsnorm_mean))).astype(np.float32)
        print('obs', self.obsnorm_mean.shape, self.obsnorm_stdev.shape)

        # Hidden layers next
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        self.hidden_layers_weights = [
            self.read_layer(layer_params[layer_name])
            for layer_name in sorted(layer_params.keys())]
        
        # Output layer
        self.out_layer_weights = self.read_layer(self.policy_params['out'])

    def call(self, obs, **kwargs):
        curr_activations_bd = (obs - self.obsnorm_mean) / (self.obsnorm_stdev + 1e-6)

        for W, b in self.hidden_layers_weights:
            curr_activations_bd = self.nonlin(tf.linalg.matmul(curr_activations_bd, W) + b)

        W, b = self.out_layer_weights
        return tf.linalg.matmul(curr_activations_bd, W) + b

    def read_layer(self, l):
        assert list(l.keys()) == ['AffineLayer']
        assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
        return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

    ##################################

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        print("\n\nThis policy class simply loads in a particular type of policy and queries it.")
        print("Not training procedure has been written, so do not try to train it.\n\n")
        raise NotImplementedError

    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None, :]
        return self.call(observation)
