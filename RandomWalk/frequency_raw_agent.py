import numpy as np


class FrequencyRawRandomAgent():
    """
    Q(sigma) agent that uses an equiprobable random policy
        POLICY:
        -ACTION 0: 0.5 probability
        -ACTION 1: 0.5 probability
        SIGMA:
        -FREQUENCY DYNAMIC: starts at 1, and modified after every episode based on the frequency distribution and sigma_factor
    """
    def __init__(self, n, alpha, gamma, sigma, sigma_factor):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma: Starting degree of sampling
        :param sigma_factor: Multiplicative factor used to reduce sigma
        """
        self.num_states = 19  # Number of states in the environment
        self.num_actions = 2  # Number of actions that the agent can take
        self.prob_left = 0.5  # Probability of moving left
        self.ep_num = 0  # Episode number tracker
        self.sigma = np.full(self.num_states, sigma, dtype=float)  # Starting degree of sampling

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.sigma_factor = sigma_factor  # Multiplicative factor that is used to reduce maximum sigma

        self.q = None  # Estimates of the reward for each state-action pair
        self.state_distr = None  # Number of times a state is visited

        self.prev_state = None  # Previous state the agent was in
        self.prev_action = None  # Previous action the agent took

    def agent_reset(self):
        """
        Resets the action-value function and state distribution of the agent
        """
        # Range of -0.5 to 0.5
        self.q = np.random.rand(self.num_states, self.num_actions) - 0.5
        self.state_distr = np.full(self.num_states, 0)

    def agent_start(self, state):
        """
        Starts the agent in the environment and makes an action
        :param state: Starting state (based on the environment)
        :return: Action the agent takes (index), sigma for the state-action pair
        """
        # Set previous state as starting state
        self.prev_state = state

        # Choose action
        self.prev_action = self.make_action()

        # Update state-action distribution
        self.state_distr[self.prev_state] += 1

        return self.prev_action, self.sigma[self.prev_state]

    def make_action(self):
        """
        Determines the action that the agent takes (based on a policy)
        :return: Action the agent takes (index)
        """
        # Equiprobable random policy
        action = 0 if np.random.uniform(0, 1) < self.prob_left else 1
        return action

    def agent_step(self, state):
        """
        Takes another step in the environment by taking an action
        :param state: Current state the agent is in
        :return: Action the agent takes (index), sigma for the state-action pair
        """
        # Choose next action
        action = self.make_action()

        # Update state and action
        self.prev_state = state
        self.prev_action = action

        # Update state distribution
        self.state_distr[self.prev_state] += 1

        return self.prev_action, self.sigma[self.prev_state]

    def agent_end(self):
        """
        Signals the end of the episode for an agent.
        """
        self.ep_num += 1

        # Calculate current frequency distribution
        tot_visits = np.sum(self.state_distr)
        freq_distr = self.state_distr / tot_visits

        # Set sigma according to raw frequency distribution
        self.sigma = (1 - freq_distr) * np.power(self.sigma_factor, self.ep_num)
