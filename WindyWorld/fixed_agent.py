import numpy as np


class FixedRandomAgent():
    """
    Q(sigma) agent that uses an equiprobable random policy
        POLICY:
        -EPSILON-GREEDY: 0.1 epsilon
        SIGMA:
        -CONSTANT: for all time
    """
    def __init__(self, n, alpha, gamma, sigma, epsilon):
        """
        :param n: Number of steps used in update
        :param alpha: Step size
        :param gamma: Discount factor
        :param sigma: Degree of sampling
        """
        self.num_rows = 7  # Number of y coordinates in the environment
        self.num_cols = 10  # Number of x coordinates in the environment
        self.num_actions = 4  # Number of actions that the agent can take

        self.n = n  # Number of steps
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discount factor
        self.sigma = sigma  # Degree of sampling
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
        action_prob = np.random.uniform(0, 1)
        greedy_actions = self.choose_greedy(state)

        if action_prob <= self.epsilon:  # Take a random action
            action = np.random.randint(self.num_actions)
        else:  # Take a greedy action
            greedy_index = np.random.randint(0, len(greedy_actions))
            action = greedy_actions[greedy_index]

        # Determine policy for the current state
        for i in range(self.num_actions):
            if i not in greedy_actions:
                pi[i] = self.epsilon / self.num_actions
            else:
                pi[i] = (1-self.epsilon)/len(greedy_actions) + self.epsilon/self.num_actions

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
        pass
