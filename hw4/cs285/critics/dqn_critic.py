from cs285.critics.base_critic import BaseCritic


class DQNCritic(BaseCritic):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        ob_dim = hparams['ob_dim']
        if isinstance(ob_dim, int):
            obs_shape = (ob_dim,)
        else:
            obs_shape = hparams['input_shape']

        self.double_q = hparams['double_q']

        q_func = hparams['q_func']
        self.q_t_model = q_func(obs_shape, hparams['ac_dim'], 'q_t_model')
        self.q_t_target = q_func(obs_shape, hparams['ac_dim'], 'q_t_target')

    def call(self, inputs, training=None, mask=None):
        return self.q_t_model(inputs, training=training, mask=mask)
