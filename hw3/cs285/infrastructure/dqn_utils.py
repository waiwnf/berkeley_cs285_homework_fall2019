"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import random
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

from cs285.infrastructure.atari_wrappers import wrap_deepmind

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def get_env_kwargs(env_name):
    pong_num_timesteps = int(2e8)
    lunar_lander_timesteps = 500000
    kwargs_by_env_name = {
        'PongNoFrameskip-v4': {
            'learning_starts': 50000,
            'target_update_freq': 10000,
            'replay_buffer_size': int(1e6),
            'num_timesteps': pong_num_timesteps,
            'q_func': atari_model,
            'learning_freq': 4,
            'grad_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'frame_history_len': 4,
            'gamma': 0.99,
            'optimizer_spec': atari_optimizer(pong_num_timesteps),
            'exploration_schedule': atari_exploration_schedule(pong_num_timesteps)
        },
        'LunarLander-v2': {
            'optimizer_spec': lander_optimizer(),
            'q_func': lander_model,
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': lunar_lander_timesteps,
            'env_wrappers': lambda env: env,
            'exploration_schedule': lander_exploration_schedule(lunar_lander_timesteps),
        }
    }
    return kwargs_by_env_name[env_name]


def lander_model(obs_shape, num_actions, name):
    hidden_dims = 64
    layers = [
        tf.keras.layers.Dense(hidden_dims, activation=tf.keras.activations.relu, input_shape=obs_shape, name='dense0'),
        tf.keras.layers.Dense(hidden_dims, activation=tf.keras.activations.relu, name='dense1'),
        tf.keras.layers.Dense(num_actions, name='dense2')]
    return tf.keras.Sequential(layers, name=name)


class IntImageModel(tf.keras.Model):
    def __init__(self, oracle):
        super().__init__()
        self.oracle = oracle

    def call(self, input, **kwargs):
        input_normalized = tf.cast(input, dtype=tf.float32) / 255.0
        return self.oracle(input_normalized)


def atari_model(obs_shape, num_actions, name):
    layers = [tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.keras.activations.relu,
                                     name='conv0'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.keras.activations.relu,
                                     name='conv1'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.keras.activations.relu,
                                     name='conv2'),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(512, activation=tf.keras.activations.relu, name='dense0'),
              tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.relu, name='dense1')]
    return IntImageModel(oracle=tf.keras.Sequential(layers, name=name))


def atari_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_ram_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_optimizer(num_timesteps):
    num_iterations = num_timesteps / 4
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    return OptimizerSpec(
        constructor=tf.keras.optimizers.Adam,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )


def lander_optimizer():
    return OptimizerSpec(
        constructor=tf.keras.optimizers.Adam,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named {!r}".format(classname))


class MemoryOptimizedReplayBuffer(object):
    def __init__(self, size, frame_history_len, obs_dtype=np.float32):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the typical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.obs_dtype = obs_dtype

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = np.zeros([self.size], dtype=np.int32) # action[i] is the action taken in state with obs[i]
        self.reward = np.zeros([self.size], dtype=np.float32) # reward[i] is received after action[i]
        # Initialize all the terminal indicators to true so that we do not go past the
        # history length in sampling.
        # done[i] is whether the rollout has terminated after action[i]
        self.done = np.ones(shape=(self.size,), dtype=np.float32)

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.stack([self._encode_observation(idx) for idx in idxes], axis=0)
        act_batch = self.action[idxes, ...]
        rew_batch = self.reward[idxes, ...]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1) for idx in idxes], axis=0)
        done_mask = self.done[idxes, ...]

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = random.sample(range(self.num_in_buffer - 1), k=batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        # Take the last frame_history_len frame indices, most recent to least recent order.
        frame_indices_reversed = [i % self.size for i in range(idx, idx - self.frame_history_len, -1)]
        # Find the most recent done marker among the selected frame indices.
        for k, frame_idx in enumerate(frame_indices_reversed[1:], start=1):
            if self.done[frame_idx] != 0:
                # Reached the "done" marker of a previous rollout. Cut off the frames history from this point.
                frame_indices_reversed = frame_indices_reversed[:k]
                break
        frame_indices = list(reversed(frame_indices_reversed))
        time_padding_length = self.frame_history_len - len(frame_indices)
        frames_history = self.obs[frame_indices, ...]
        if time_padding_length > 0:
            time_padding = np.zeros((time_padding_length,) + self.obs.shape[1:], dtype=self.obs.dtype)
            frames_history = np.concatenate((time_padding, frames_history), axis=0)
        img_h, img_w = self.obs.shape[1], self.obs.shape[2]
        # Concatenate the time steps and color channels into a single dimension per pixel.
        return np.transpose(frames_history, axes=(1, 2, 0, 3)).reshape((img_h, img_w, -1))

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=self.obs_dtype)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
        self.obs[self.next_idx, ...] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after observing frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that one can call `encode_recent_observation`
        in between.

        Parameters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
