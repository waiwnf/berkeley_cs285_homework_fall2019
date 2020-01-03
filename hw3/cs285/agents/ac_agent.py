import numpy as np
import tensorflow as tf

from collections import OrderedDict

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import DiscreteMLPPolicy, ContinuousMLPPolicy
from cs285.critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params, **kwargs):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        # actor/policy
        if self.agent_params['discrete']:
            self.actor = DiscreteMLPPolicy(self.agent_params['ac_dim'],
                                           self.agent_params['ob_dim'],
                                           self.agent_params['n_layers'],
                                           self.agent_params['size'])
        else:
            self.actor = ContinuousMLPPolicy(self.agent_params['ac_dim'],
                                             self.agent_params['ob_dim'],
                                             self.agent_params['n_layers'],
                                             self.agent_params['size'])
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.agent_params['learning_rate'])

        self.critic = BootstrappedContinuousCritic(self.agent_params)
        self.critic_loss = tf.keras.losses.MeanSquaredError()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.agent_params['learning_rate'])
        self.critic.nn_critic.compile(optimizer=self.critic_optimizer, loss=self.critic_loss)

        self.replay_buffer = ReplayBuffer()

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):

        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        current_state_values = self.critic(ob_no)
        next_state_values = self.gamma * self.critic(next_ob_no) * (1.0 - tf.expand_dims(tf.cast(terminal_n, tf.float32), axis=1))
        adv_n = next_state_values + re_n - current_state_values

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            for _ in range(self.agent_params['num_target_updates']):
                critic_targets = self.critic.get_training_targets(next_ob_no, re_n, terminal_n, self.gamma)
                critic_dataset = tf.data.Dataset.from_tensor_slices(
                    (tf.cast(ob_no, tf.float32), tf.cast(critic_targets, tf.float32)))
                critic_dataset = critic_dataset.batch(batch_size=critic_targets.shape[0]).repeat()
                self.critic.nn_critic.fit(critic_dataset, epochs=1,
                                          steps_per_epoch=self.agent_params['num_grad_steps_per_target_update'])

        advantage = tf.stop_gradient(self.estimate_advantage(ob_no, next_ob_no, tf.expand_dims(re_n, axis=1), terminal_n))

        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            with tf.GradientTape() as tape:
                log_action_probas = self.actor.get_log_prob(ob_no, ac_na)
                loss = -tf.reduce_mean(advantage * tf.expand_dims(log_action_probas, axis=1))

            actor_vars = self.actor.trainable_variables
            grads = tape.gradient(loss, actor_vars)
            self.policy_optimizer.apply_gradients(zip(grads, actor_vars))

        loss_dict = OrderedDict()
        loss_dict['Critic_Loss'] = 0  # put final critic loss here
        loss_dict['Actor_Loss'] = loss.numpy().item()  # put final actor loss here
        return loss_dict

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
