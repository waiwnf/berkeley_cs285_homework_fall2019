import tensorflow as tf


class BaseCritic(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        raise NotImplementedError
