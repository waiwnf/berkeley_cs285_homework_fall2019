import tensorflow as tf

from cs285.agents.base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer


class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'])  # TODO: look in here and implement this

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.agent_params['learning_rate'])

        self.actor.compile(optimizer=self.optimizer, loss=self.loss)

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train_multi_iter(self, batch_size, num_iters):
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(self.replay_buffer.obs, tf.float32), tf.cast(self.replay_buffer.acs, tf.float32)))
        dataset = dataset.shuffle(self.replay_buffer.obs.shape[0])
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).repeat()
        self.actor.fit(dataset, epochs=1, steps_per_epoch=num_iters)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        with tf.GradientTape() as tape:
            pred_actions = self.actor(ob_no)
            loss_value = self.loss(ac_na, pred_actions)
        trainable_vars = self.actor.trainable_variables
        grads = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss_value

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size) ## TODO: look in here and implement this
