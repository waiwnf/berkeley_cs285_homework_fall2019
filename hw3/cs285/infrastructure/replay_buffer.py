import numpy as np

from cs285.infrastructure.utils import get_pathlength, convert_listofrollouts


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths):

        # add new rollouts into our list of rollouts
        self.paths.extend(paths)

        # Remove the old rollouts beyond the self.max_size observations
        # so that they can be garbage collected and the memory consumption
        # does not grow indefinitely.
        path_lengths = [get_pathlength(path) for path in self.paths]
        total_steps = 0
        first_retained_path_idx = 0
        for i in range(len(path_lengths) - 1, -1, -1):
            total_steps += path_lengths[i]
            if total_steps >= self.max_size:
                first_retained_path_idx = i
                break
        self.paths = self.paths[first_retained_path_idx:]

        # convert new rollouts into their component arrays, and append them onto our arrays
        (observations, actions, next_observations, terminals, concatenated_rews,
         unconcatenated_rews) = convert_listofrollouts(paths)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
            self.unconcatenated_rews = unconcatenated_rews
        else:
            # Make copies of the updated arrays to avoid holding the whole slice bases in memory.
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:].copy()
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:].copy()
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:].copy()
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:].copy()
            self.concatenated_rews = np.concatenate([self.concatenated_rews, concatenated_rews])[-self.max_size:].copy()
            if isinstance(unconcatenated_rews, list):
                self.unconcatenated_rews += unconcatenated_rews
            else:
                self.unconcatenated_rews.append(unconcatenated_rews)

        self.unconcatenated_rews = self.unconcatenated_rews[first_retained_path_idx:]

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert (self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] ==
                self.terminals.shape[0])
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return (self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices],
                self.next_obs[rand_indices], self.terminals[rand_indices])

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return (self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:],
                    self.next_obs[-batch_size:], self.terminals[-batch_size:])
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -= 1
                num_recent_rollouts_to_return += 1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            (observations, actions, next_observations, terminals, concatenated_rews,
             unconcatenated_rews) = convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals
