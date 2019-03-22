
class WalkEnvironment():
    """
    19 state 1-dimensional deterministic environment with negative reward on the far left and positive reward on the far right
        START STATE:
        -CENTER (9)
        TERMINAL STATES:
        -'STATE' -1: -1 reward
        -'STATE' 19: +1 reward
        ACTIONS:
        -ACTION 0: MOVE LEFT (-1)
        -ACTION 1: MOVE RIGHT (+1)
    """
    def __init__(self):
        self.num_states = 19
        self.start_state = 9
        self.actions = [-1, 1]  # Action space

        self.current_state = None  # State the agent is currently in

    def env_start(self):
        """
        Initialize environment variables
        """
        self.current_state = self.start_state
        return self.current_state

    def env_step(self, action):
        """
        Moves the agent through the environment
        :param action: Action the agent takes (index)
        :return: Reward from the action, current state of the agent, whether the current state is terminal
        """
        # Update current stated based on the action the agent took
        self.current_state += self.actions[action]

        # Check if the agent reached a terminal state
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
