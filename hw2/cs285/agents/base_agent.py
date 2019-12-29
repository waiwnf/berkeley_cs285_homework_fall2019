class BaseAgent(object):
    def train(self, obs, acs, rews_list, next_obs, terminals):
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError