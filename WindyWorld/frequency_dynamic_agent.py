import numpy as np


class FrequencyRandomAgent():
    """
    Q(sigma) agent that uses an equiprobable random policy
        POLICY:
        -EPSILON-GREEDY
        SIGMA:
        -FREQUENCY DYNAMIC: starts at 1, and modified after every episode based on the frequency distribution and sigma_factor
    """
    def __init__(self, n, alpha, gamma, sigma_factor, epsilon, use_mean):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma_factor: Multiplicative factor used to reduce sigma
        """
        self.num_rows = 7  # Number of y coordinates in the environment
        self.num_cols = 10  # Number of x coordinates in the environment
        self.num_actions = 4  # Number of actions that the agent can take
        self.ep_num = 0
        self.sigma = np.full((self.num_states, self.num_actions), 1, dtype=float)  # Degree of sampling
        self.sa_distr = None  # Number of times state-action pairs are visited

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.sigma_factor = sigma_factor  # Multiplicative factor that is used to reduce maximum sigma
        self.epsilon = epsilon  # Probability of agent taking a random action
        self.use_mean = use_mean  # Flag on whether to use raw frequency distribution or mean/st.d in sigma calculation

        self.prev_state = None  # Previous state the agent was in
        self.prev_action = None  # Previous action the agent took

        self.q = None  # Estimates of the reward for each action

    def agent_init(self):
        """
        Arbitrarily initializes the action-value function of the agent
        """
        # Initialize action-value function with all 0's
        self.q = np.full((self.num_rows, self.num_cols, self.num_actions), 0, dtype=float)
        self.sa_distr = np.full((self.num_rows, self.num_cols, self.num_actions), 0)

    def agent_start(self, state):
        """
        Starts the agent in the environment and makes an action
        :param state: Starting state (based on the environment)
        :return: Action the agent takes (index), policy for that state, sigma for the state-action pair
        """
        # Set previous state as starting state
        self.prev_state = state

        # Choose action
        (self.prev_action, pi) = self.make_action(state)

        # Update state-action distribution
        self.sa_distr[state[0]][state[1]][self.prev_action] += 1

        return (self.prev_action, pi, self.sigma[state[0]][state[1]][self.prev_action])

    def make_action(self, state):
        """
        Determines the action that the agent takes (using an epsilon-greedy algorithm)
        :return: Action the agent takes (index), policy for that state
        """
        pi = np.full(self.num_actions, 0, dtype=float)
        greedy_actions = self.choose_greedy(state)

        # Determine policy for the current state
        for i in range(self.num_actions):
            if i not in greedy_actions:
                pi[i] = self.epsilon / self.num_actions
            else:
                pi[i] = (1-self.epsilon)/len(greedy_actions) + self.epsilon/self.num_actions

        action = np.random.choice(range(self.num_actions), p=pi)

        return (action, pi)

    def choose_greedy(self, state):
        """
        Determines the optimal action according to q
        :param state: State the agent is currently in
        :return: Array of possible greedy actions
        """
        greedy_actions = np.ravel(np.argwhere(self.q[state[0], state[1], :] == np.max(self.q[state[0], state[1], :])))

        return greedy_actions

    def agent_step(self, state):
        """
        Takes another step in the environment by taking an action
        :param state: Current state the agent is in
        :return: Action the agent takes (index), policy for that state, sigma for the state-action pair
        """
        # Choose next action
        (action, pi) = self.make_action(state)

        # Update state and action
        self.prev_state = state
        self.prev_action = action

        # Update state-action distribution
        self.sa_distr[state[0]][state[1]][self.prev_action] += 1

        return (self.prev_action, pi, self.sigma[state[0]][state[1]][self.prev_action])

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
