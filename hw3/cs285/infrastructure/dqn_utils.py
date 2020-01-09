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
            'obs_dtype': np.uint8,
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
            'obs_dtype': np.float32,
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
              tf.keras.layers.Dense(num_actions, name='dense1')]
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


class ConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def __call__(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
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

    def __call__(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
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

    def __call__(self, t):
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
        self.action = np.zeros([self.size], dtype=np.int32)  # action[i] is the action taken in state with obs[i]
        self.reward = np.zeros([self.size], dtype=np.float32)  # reward[i] is received after action[i]
        # Initialize all the terminal indicators to true so that we do not go past the
        # history length in sampling.
        # done[i] is whether the rollout has terminated after action[i]
        self.done = np.ones(shape=(self.size,), dtype=np.float32)

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size < self.num_in_buffer

    def _get_time_sliced_frames_history(self, idx, history_len):
        # Take the last frame_history_len frame indices, most recent to least recent order.
        frame_indices_reversed = [i % self.size for i in range(idx, idx - history_len, -1)]
        # Find the most recent done marker among the selected frame indices.
        for k, frame_idx in enumerate(frame_indices_reversed[1:], start=1):
            if self.done[frame_idx] != 0:
                # Reached the "done" marker of a previous rollout. Cut off the frames history from this point.
                frame_indices_reversed = frame_indices_reversed[:k]
                break
        frame_indices = list(reversed(frame_indices_reversed))
        time_padding_length = history_len - len(frame_indices)
        frames_history = self.obs[frame_indices, ...]
        if time_padding_length > 0:
            time_padding = np.zeros((time_padding_length,) + self.obs.shape[1:], dtype=self.obs.dtype)
            frames_history = np.concatenate((time_padding, frames_history), axis=0)
        return frames_history

    @staticmethod
    def _frames_history_to_observation(frames_history):
        new_axes_order = tuple(range(1, len(frames_history.shape) - 1)) + (0, len(frames_history.shape) - 1)
        shape_new_axes_order = tuple(frames_history.shape[idx] for idx in new_axes_order)
        new_shape = shape_new_axes_order[:-2] + (shape_new_axes_order[-2] * shape_new_axes_order[-1],)
        return np.transpose(frames_history, axes=new_axes_order).reshape(new_shape)

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
        # Subtract 1 from the buffer size to make sure we have a valid next observation for every step.
        idxes = random.sample(range(self.num_in_buffer - 1), k=batch_size)

        obs_batch = np.stack([self._encode_observation(idx) for idx in idxes], axis=0)
        act_batch = self.action[idxes, ...]
        rew_batch = self.reward[idxes, ...]

        # TODO use a dummy zero observation if the current step ended with a terminal.
        next_obs_batch = np.stack([self._encode_observation(idx + 1) for idx in idxes], axis=0)
        done_mask = self.done[idxes, ...]

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def encode_next_frame_observation(self, frame):
        prev_idx = (self.next_idx - 1) % self.size
        if self.done[prev_idx]:
            history_frames = np.zeros((self.frame_history_len - 1,) + self.obs.shape[1:], dtype=self.obs.dtype)
        else:
            history_frames = self._get_time_sliced_frames_history(prev_idx, self.frame_history_len - 1)
        all_frames = np.concatenate([history_frames, frame[np.newaxis, ...]], axis=0)
        return self._frames_history_to_observation(all_frames)

    def _encode_observation(self, idx):
        # T x H x W x C
        frames_history = self._get_time_sliced_frames_history(idx, self.frame_history_len)
        return self._frames_history_to_observation(frames_history)

    def store_step(self, frame, action, reward, done):
        """Store a single transition in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        if self.obs is None:
            self.obs = np.empty((self.size,) + frame.shape, dtype=self.obs_dtype)
        self.obs[self.next_idx, ...] = frame
        self.action[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.done[self.next_idx] = done

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
