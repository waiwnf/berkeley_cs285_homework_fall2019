import tensorflow as tf
import numpy as np

from cs285.agents.base_agent import BaseAgent
from cs285.policies.MLP_policy import DiscreteMLPPolicy, ContinuousMLPPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.tf_utils import build_mlp


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params, batch_size=500000, **kwargs):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']

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

        # replay buffer
        self.replay_buffer = ReplayBuffer(2 * batch_size)

        self.baseline_model = None
        if self.agent_params['nn_baseline']:
            self.baseline_model = build_mlp((self.agent_params['ob_dim'],), output_size=1,
                                            n_layers=self.agent_params['n_layers'], size=self.agent_params['size'],
                                            name='baseline_model')
            self.baseline_loss = tf.keras.losses.MeanSquaredError()
            self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.agent_params['learning_rate'])
            self.baseline_model.compile(optimizer=self.baseline_optimizer, loss=self.baseline_loss)

    def train(self, obs, acs, rews_list, next_obs, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.

            ---------------------------------------------------------------------------------- 
            
            Recall that the expression for the policy gradient PG is
            
                PG = E_{tau} [sum_{t=0}^{T-1} grad log pi(a_t|s_t) * (Q_t - b_t )]
            
                where 
                tau=(s_0, a_0, s_1, a_1, s_2, a_2, ...) is a trajectory,
                Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                b_t is a baseline which may depend on s_t,
                and (Q_t - b_t ) is the advantage.

            Thus, the PG update performed by the actor needs (s_t, a_t, q_t, adv_t),
                and that is exactly what this function provides.

            ----------------------------------------------------------------------------------
        """

        # step 1: calculate q values of each (s_t, a_t) point, 
        # using rewards from that full rollout of length T: (r_0, ..., r_t, ..., r_{T-1})
        q_values = self.calculate_q_vals(rews_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantage_values = self.estimate_advantage(obs, q_values)

        # step 3:
        # TODO: pass the calculated values above into the actor/policy's update, 
        # which will perform the actual PG update step

        # TODO: define the loss that should be optimized when training a policy with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: see define_log_prob (above)
        # to get log pi(a_t|s_t)
        # HINT3: look for a placeholder above that will be populated with advantage values 
        # to get [Q_t - b_t]
        # HINT4: don't forget that we need to MINIMIZE this self.loss
        # but the equation above is something that should be maximized

        # define the log probability of seen actions/observations under the current policy
        with tf.GradientTape() as tape:
            log_action_probas = self.actor.get_log_prob(obs, acs)
            advantage_values_no_grad = tf.stop_gradient(advantage_values)
            loss = -tf.reduce_mean(advantage_values_no_grad * log_action_probas)

        actor_vars = self.actor.trainable_variables
        grads = tape.gradient(loss, actor_vars)
        self.policy_optimizer.apply_gradients(zip(grads, actor_vars))

        if self.nn_baseline:
            targets_n = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-8)
            dataset = tf.data.Dataset.from_tensor_slices(
                (tf.cast(obs, tf.float32), tf.cast(targets_n, tf.float32)))
            dataset = dataset.batch(batch_size=targets_n.shape[0]).repeat()
            # 20 baseline gradient updates with the current data batch.
            self.baseline_model.fit(dataset, epochs=1, steps_per_epoch=20)

        return loss.numpy().item()

    def calculate_q_vals(self, rews_list):

        """
            Monte Carlo estimation of the Q function.

            arguments:
                rews_list: length: number of sampled rollouts
                    Each element corresponds to a particular rollout,
                    and contains an array of the rewards for every step of that particular rollout

            returns:
                q_values: shape: (sum/total number of steps across the rollouts)
                    Each entry corresponds to the estimated q(s_t,a_t) value 
                    of the corresponding obs/ac point at time t.
 
        """

        # Case 1: trajectory-based PG 
        if not self.reward_to_go:

            # TODO: Estimate the Q value Q^{pi}(s_t, a_t) using rewards from that entire trajectory
            # HINT1: value of each point (t) = total discounted reward summed over the entire trajectory (from 0 to T-1)
            # In other words, q(s_t, a_t) = sum_{t'=0}^{T-1} gamma^t' r_{t'}
            # Hint3: see the helper functions at the bottom of this file
            q_values = np.concatenate([self._discounted_return(r) for r in rews_list])

        # Case 2: reward-to-go PG 
        else:

            # TODO: Estimate the Q value Q^{pi}(s_t, a_t) as the reward-to-go
            # HINT1: value of each point (t) = total discounted reward summed over the remainder of that trajectory
            # (from t to T-1)
            # In other words, q(s_t, a_t) = sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            # Hint3: see the helper functions at the bottom of this file
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rews_list])

        return q_values.astype(np.float32)

    def estimate_advantage(self, obs, q_values):

        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """

        # TODO: Estimate the advantage when nn_baseline is True
        # HINT1: pass obs into the neural network that you're using to learn the baseline
        # extra hint if you're stuck: see your actor's run_baseline_prediction
        # HINT2: advantage should be [Q-b]
        if self.nn_baseline:
            b_n_unnormalized = self.baseline_model(obs)
            b_n = b_n_unnormalized * np.std(q_values) + np.mean(q_values)
            adv_n = (q_values - tf.squeeze(b_n)).numpy()
        # Else, just set the advantage to [Q]
        else:
            adv_n = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return adv_n.astype(np.float32)

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^{T-1} gamma^t' r_{t'}
                note that all entries of this output are equivalent
                because each index t is a sum from 0 to T-1 (and doesnt involve t)
        """

        q = sum(reward * (self.gamma ** t) for t, reward in enumerate(rewards))
        return [q for _ in rewards]

    def _discounted_cumsum(self, rewards):
        """
            Input:
                a list of length T 
                a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T
            Output: 
                a list of length T
                a list where the entry in each index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """

        all_discounted_cumsums = rewards.copy()
        for t in range(len(all_discounted_cumsums) - 1, 0, -1):
            all_discounted_cumsums[t - 1] += self.gamma * all_discounted_cumsums[t]
        return all_discounted_cumsums
