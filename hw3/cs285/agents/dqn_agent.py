import random

import tensorflow as tf
import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params, **kwargs):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()
        self.total_episode_reward = 0.0
        self.total_episodes = []
        self.episode_num = 0

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.gamma = agent_params['gamma']

        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params)
        self.q_t_loss = tf.keras.losses.Huber()
        self.q_t_optimizer = self.optimizer_spec.constructor(clipnorm=agent_params['grad_norm_clipping'],
                                                             learning_rate=self.optimizer_spec.lr_schedule,
                                                             **self.optimizer_spec.kwargs)

        self.actor = ArgMaxPolicy(self.critic)

        self.replay_buffer = MemoryOptimizedReplayBuffer(agent_params['replay_buffer_size'],
                                                         agent_params['frame_history_len'],
                                                         obs_dtype=agent_params['obs_dtype'])
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):

        """
            Step the env and store the transition

            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.

            Note that self.last_obs must always point to the new latest observation.
        """

        eps = self.exploration(self.t)
        # TODO use epsilon greedy exploration when selecting action
        # HINT: take random action 
        # with probability eps (see np.random.random())
        # OR if your current step number (see self.t) is less that self.learning_starts
        perform_random_action = random.random() < eps

        if perform_random_action:
            action = random.randrange(self.num_actions)
        else:
            # TODO query the policy to select action
            # HINT: you cannot use "self.last_obs" directly as input
            # into your network, since it needs to be processed to include context
            # from previous frames. 
            # Check out the replay buffer, which has a function called
            # encode_recent_observation that will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            enc_last_obs = self.replay_buffer.encode_next_frame_observation(self.last_obs)[np.newaxis, ...]

            # TODO query the policy with enc_last_obs to select action
            action = self.actor.get_action(enc_last_obs).numpy().item()

        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
        # obs, reward, done, info = env.step(action)
        prev_obs = self.last_obs
        self.last_obs, reward, env_done, _ = self.env.step(action)
        self.total_episode_reward += reward

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see replay buffer's store_effect function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_step(prev_obs, action, reward, env_done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if env_done:
            self.episode_num += 1
            print('Total episode {}: {}'.format(self.episode_num, self.total_episode_reward))
            self.last_obs = self.env.reset()
            self.total_episodes.append(self.total_episode_reward)
            self.total_episode_reward = 0.0

    def sample(self, batch_size):
        return None, None, None, None, None

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        """
            Here, you should train the DQN agent.
            This consists of training the critic, as well as periodically updating the target network.
        """

        loss = 0.0
        if ((self.t > self.learning_starts) and (self.t % self.learning_freq == 0) and (
        self.replay_buffer.can_sample(self.batch_size))):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = (
                tf.convert_to_tensor(x) for x in self.replay_buffer.sample(self.batch_size))
            next_state_target_q_a = self.critic.q_t_target(next_obs_batch)

            with tf.GradientTape() as tape:
                if self.critic.double_q:
                    next_state_q_a = self.critic.q_t_model(next_obs_batch)
                    next_actions = tf.argmax(next_state_q_a, axis=1)
                else:
                    next_actions = tf.argmax(next_state_target_q_a, axis=1)
                next_state_actions_mask = tf.one_hot(next_actions, depth=self.num_actions)
                q_target = rew_batch + self.gamma * tf.reduce_sum(
                    next_state_target_q_a * next_state_actions_mask, axis=1) * (1.0 - done_mask)
                q_target = tf.stop_gradient(q_target)
                current_state_q_a = self.critic.q_t_model(obs_batch)
                pred_q = tf.reduce_sum(current_state_q_a * tf.one_hot(act_batch, depth=self.num_actions), axis=1)
                loss_value = self.q_t_loss(q_target, pred_q)

            trainable_vars = self.critic.q_t_model.trainable_variables
            grads = tape.gradient(loss_value, trainable_vars)
            self.q_t_optimizer.apply_gradients(zip(grads, trainable_vars))
            self.num_param_updates += 1
            loss = loss_value.numpy().item()

            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.q_t_target.set_weights(self.critic.q_t_model.get_weights())

        self.t += 1
        return loss
