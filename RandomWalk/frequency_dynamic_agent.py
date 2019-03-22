import numpy as np


class FrequencyRandomAgent():
    """
    Q(sigma) agent that uses an equiprobable random policy
        POLICY:
        -ACTION 0: 0.5 probability
        -ACTION 1: 0.5 probability
        SIGMA:
        -FREQUENCY DYNAMIC: starts at 1, and modified after every episode based on the frequency distribution and sigma_factor
    """
    def __init__(self, n, alpha, gamma, sigma_factor):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma_factor: Multiplicative factor used to reduce sigma
        """
        self.num_states = 19  # Number of states in the environment
        self.num_actions = 2  # Number of actions that the agent can take
        self.prob_left = 0.5  # Probability of moving left
        self.sigma = np.full((self.num_states, self.num_actions), 1, dtype=float)  # Degree of sampling
        self.sigma_factor = sigma_factor
        self.sa_distr = None  # Number of times state-action pairs are visited

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.ep_num = 0

        self.prev_state = None  # Previous state the agent was in
        self.prev_action = None  # Previous action the agent took

        self.q = None  # Estimates of the reward for each action

    def agent_init(self):
        """
        Arbitrarily initializes the action-value function of the agent
        """
        # Range of -0.5 to 0.5
        self.q = np.random.rand(self.num_states, self.num_actions) - 0.5
        self.sa_distr = np.full((self.num_states, self.num_actions), 0)

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
        self.sa_distr[self.prev_state][self.prev_action] += 1

        return self.prev_action, self.sigma[self.prev_state][self.prev_action]

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

        # Update state-action distribution
        self.sa_distr[self.prev_state][self.prev_action] += 1

        return self.prev_action, self.sigma[self.prev_state][self.prev_action]

    def agent_end(self):
        """
        Signals the end of the episode for an agent.
        """
        tot_visits = np.sum(self.sa_distr)
        freq_distr = self.sa_distr / tot_visits
        self.sigma = (1 - freq_distr) * np.power(self.sigma_factor, self.ep_num)
        self.ep_num += 1
