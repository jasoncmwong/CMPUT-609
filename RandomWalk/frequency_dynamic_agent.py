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
    def __init__(self, n, alpha, gamma, sigma_factor, use_mean):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma_factor: Multiplicative factor used to reduce sigma
        """
        self.num_states = 19  # Number of states in the environment
        self.num_actions = 2  # Number of actions that the agent can take
        self.prob_left = 0.5  # Probability of moving left
        self.ep_num = 0
        self.sigma = np.full((self.num_states, self.num_actions), 1, dtype=float)  # Degree of sampling
        self.sa_distr = None  # Number of times state-action pairs are visited

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.sigma_factor = sigma_factor  # Multiplicative factor that is used to reduce maximum sigma
        self.use_mean = use_mean  # Flag on whether to use raw frequency distribution or mean/st.d in sigma calculation

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
        self.ep_num += 1

        # Calculate current frequency distribution
        tot_visits = np.sum(self.sa_distr)
        freq_distr = self.sa_distr / tot_visits

        if self.use_mean:  # Use mean and standard deviation thresholds
            # Determine thresholds for sigma extremes
            freq_mean = np.mean(freq_distr)
            freq_std = np.std(freq_distr)
            up_thresh = (freq_mean + freq_std) * np.power(self.sigma_factor, self.ep_num)
            bot_thresh = (freq_mean - freq_std) * np.power(self.sigma_factor, self.ep_num)

            # Set sigma according to thresholds
            for i in range(self.num_states):
                for j in range(self.num_actions):
                    curr_freq = freq_distr[i][j]
                    if curr_freq > up_thresh:  # Visited enough - perform expectation updates
                        self.sigma[i][j] = 0
                    elif curr_freq < bot_thresh:  # Not visited enough - perform full sampling
                        self.sigma[i][j] = 1
                    else:  # In the middle - use intermediate sigma
                        self.sigma[i][j] = 1 - (curr_freq - bot_thresh)/(2*freq_std*np.power(self.sigma_factor, self.ep_num))
        else:  # Use the raw frequency distribution reduced by the multiplicative factor
            self.sigma = (1 - freq_distr) * np.power(self.sigma_factor, self.ep_num)
