import numpy as np


class EpisodicAgent:
    """
    Q(sigma) agent that uses an equiprobable random policy
        POLICY:
        -EPSILON-GREEDY
        SIGMA:
        -EPISODE DYNAMIC: starts at sigma, and reduced by a factor of sigma_factor every episode
    """
    def __init__(self, n, alpha, gamma, sigma, sigma_factor, epsilon):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma: Starting degree of sampling
        :param sigma_factor: Multiplicative factor used to reduce sigma
        """
        self.num_rows = 7  # Number of y coordinates in the environment
        self.num_cols = 10  # Number of x coordinates in the environment
        self.num_actions = 4  # Number of actions that the agent can take
        self.ep_num = 0  # Episode number tracker

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.sigma = np.full((self.num_rows, self.num_cols), sigma, dtype=float)  # Starting degree of sampling
        self.sigma_factor = sigma_factor  # Multiplicative factor that is used to reduce maximum sigma
        self.epsilon = epsilon  # Probability of agent taking a random action

        self.q = None  # Estimates of the reward for each state-action pair

        self.prev_state = None  # Previous state the agent was in
        self.prev_action = None  # Previous action the agent took

    def agent_reset(self):
        """
        Resets the action-value function of the agent
        """
        self.q = np.full((self.num_rows, self.num_cols, self.num_actions), 0, dtype=float)

    def agent_start(self, state):
        """
        Starts the agent in the environment and makes an action
        :param state: Starting state (based on the environment)
        :return: Action the agent takes (index), policy for that state, sigma for the state
        """
        # Set previous state as starting state
        self.prev_state = state

        # Choose action
        (self.prev_action, pi) = self.make_action(state)

        return self.prev_action, pi, self.sigma[self.prev_state[0]][self.prev_state[1]]

    def make_action(self, state):
        """
        Determines the action that the agent takes (using an epsilon-greedy algorithm)
        :param state: State the agent is currently in
        :return: Action the agent takes (index), policy for the state
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

        return action, pi

    def choose_greedy(self, state):
        """
        Determines the optimal action according to q
        :param state: Current state the agent is in
        :return: Array of possible greedy actions
        """
        greedy_actions = np.ravel(np.argwhere(self.q[state[0], state[1], :] == np.max(self.q[state[0], state[1], :])))

        return greedy_actions

    def agent_step(self, state):
        """
        Takes another step in the environment by taking an action
        :param state: Current state the agent is in
        :return: Action the agent takes (index), policy for the state, sigma for the state-action pair
        """
        # Choose next action
        (action, pi) = self.make_action(state)

        # Update state and action
        self.prev_state = state
        self.prev_action = action

        return self.prev_action, pi, self.sigma[self.prev_state[0]][self.prev_state[1]]

    def agent_end(self):
        """
        Signals the end of the episode for an agent.
        """
        # Reduce sigma by a constant factor
        self.sigma *= self.sigma_factor
