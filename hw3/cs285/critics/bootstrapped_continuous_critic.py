from cs285.critics.base_critic import BaseCritic
import tensorflow as tf
from cs285.infrastructure.tf_utils import build_mlp


class BootstrappedContinuousCritic(BaseCritic):
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']

        self.nn_critic = build_mlp((hparams['ob_dim'],), output_size=1, n_layers=hparams['n_layers'], size=hparams['size'], name='nn_critic')

    def call(self, obs, **kwargs):
        return self.nn_critic(obs)

    def get_training_targets(self, next_observations, rewards, terminals, gamma):
        # recompute the target values by
        # a) calculating V(s') by querying this critic network (ie calling 'forward') with next_ob_no
        # b) and computing the target values as r(s, a) + gamma * V(s')
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it to 0) when a terminal state is reached
        raw_next_values = self.call(next_observations)
        terminals_value_mask = 1.0 - tf.expand_dims(tf.cast(terminals, tf.float32), axis=1)
        next_state_values = raw_next_values * terminals_value_mask
        return tf.expand_dims(rewards, axis=1) + gamma * next_state_values

    # def update(self, ob_no, next_ob_no, re_n, terminal_n):
    #     """
    #         Update the parameters of the critic.
    #
    #         let sum_of_path_lengths be the sum of the lengths of the sampled paths
    #         let num_paths be the number of sampled paths
    #
    #         arguments:
    #             ob_no: shape: (sum_of_path_lengths, ob_dim)
    #             next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
    #             re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
    #                 the reward for each timestep
    #             terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
    #                 at that timestep of 0 if the episode did not end
    #
    #         returns:
    #             loss
    #     """
    #
    #     # TODO: Implement the pseudocode below:
    #
    #     # do the following (self.num_grad_steps_per_target_update * self.num_target_updates) times:
    #         # every self.num_grad_steps_per_target_update steps (which includes the first step),
    #             # recompute the target values by
    #                 #a) calculating V(s') by querying this critic network (ie calling 'forward') with next_ob_no
    #                 #b) and computing the target values as r(s, a) + gamma * V(s')
    #             # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it to 0) when a terminal state is reached
    #         # every time,
    #             # update this critic using the observations and targets
    #             # HINT1: need to sess.run the following:
    #                 #a) critic_update_op
    #                 #b) critic_loss
    #             # HINT2: need to populate the following (in the feed_dict):
    #                 #a) sy_ob_no with ob_no
    #                 #b) sy_target_n with target values calculated above
    #
    #     TODO
    #
    #     return loss
