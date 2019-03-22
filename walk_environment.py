
class WalkEnvironment():
    """
    Example 1-Dimensional environment
    """

    def __init__(self):
        """Declare environment variables."""
        self.num_states = 19
        self.start_state = 9
        self.actions = [-1, 1]

        # state we are in currently
        self.current_state = None

    def env_start(self):
        """
        Initialize environment variables.
        """
        self.current_state = self.start_state
        return self.current_state

    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        # action = -1 for left; +1 for right
        self.current_state += self.actions[action]

        # This environment will give a +1 reward if the agent terminates on
        # the right, otherwise 0 reward
        if self.current_state == self.num_states:  # Far right of environment
            is_terminal = True
            reward = 1.0
        elif self.current_state == -1:  # Far left of environment
            is_terminal = True
            reward = -1.0
        else:
            is_terminal = False
            reward = 0.0

        return reward, self.current_state, is_terminal
