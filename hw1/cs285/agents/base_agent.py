class BaseAgent(object):
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        raise NotImplementedError()

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()