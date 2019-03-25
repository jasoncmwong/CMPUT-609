import numpy as np


class EpisodicRandomAgent():
    """
    Q(sigma) agent that uses an equiprobable random policy
        POLICY:
        -ACTION 0: 0.5 probability
        -ACTION 1: 0.5 probability
        SIGMA:
        -EPISODE DYNAMIC: starts at 1, and reduced by a factor of sigma_factor every episode
    """
    def __init__(self, n, alpha, gamma, sigma_factor, epsilon):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma_factor: Multiplicative factor used to reduce sigma
        """
        self.num_rows = 7  # Number of y coordinates in the environment
        self.num_cols = 10  # Number of x coordinates in the environment
        self.num_actions = 4  # Number of actions that the agent can take
        self.sigma = 1  # Degree of sampling

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.sigma_factor = sigma_factor
        self.epsilon = epsilon  # Probability of choosing a random action

        self.prev_state = None  # Previous state the agent was in
        self.prev_action = None  # Previous action the agent took

        self.q = None  # Estimates of the reward for each action

    def agent_init(self):
        """
        Arbitrarily initializes the action-value function of the agent
        """
        # Initialize action-value function with all 0's
        self.q = np.full((self.num_rows, self.num_cols, self.num_actions), 0, dtype=float)

    def agent_start(self, state):
        """
        Starts the agent in the environment and makes an action
        :param state: Starting state (based on the environment)
        :return: Action the agent takes (index), probability of taking that action
        """
        # Set previous state as starting state
        self.prev_state = state

        # Choose action
        (self.prev_action, pi) = self.make_action(state)

        return (self.prev_action, pi)

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
        :return: Action the agent takes (index), policy for that state
        """
        # Choose next action
        (action, pi) = self.make_action(state)

        # Update state and action
        self.prev_state = state
        self.prev_action = action

        return (self.prev_action, pi)

    def agent_end(self):
        """
        Signals the end of the episode for an agent.
        """
        # Reduce sigma by a constant factor
        self.sigma *= self.sigma_factor
