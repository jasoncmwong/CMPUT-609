
class WindyEnvironment():
    """
    2-dimensional grid world (7 rows x 10 columns) with a wind affecting certain columns that pushes the agent up; agent must navigate to a goal past the wind
        START STATE:
        -MIDDLE, FAR LEFT (3, 0)
        TERMINAL STATE:
        -(3, 7): 0 reward (-1 on each step otherwise)
        WIND:
        -COLUMNS {3, 4, 5, 8}: 1
        -COLUMNS {6, 7}: 2
        ACTIONS:
        -ACTION 0: MOVE DOWN [-1, 0]
        -ACTION 1: MOVE LEFT [0, -1]
        -ACTION 2: MOVE UP [1, 0]
        -ACTION 3: MOVE RIGHT [0, 1]
    """
    def __init__(self):
        self.num_rows = 7  # Number of rows in the grid world (y-direction)
        self.num_cols = 10  # Number of columns in the grid world (x-direction)
        self.start_state = [3, 0]  # State the agent is in at the start of an episode
        self.terminal_state = [3, 7]  # Terminal state of the environment
        self.actions = [  # Action space
            [-1, 0],    # Move down
            [0, -1],    # Move left
            [1, 0],     # Move up
            [0, 1]      # Move right
            ]
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # Wind values for each column of the grid world

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
        curr_x = self.current_state[1]
        self.current_state[0] += self.actions[action][0] + self.wind[curr_x]
        self.current_state[1] += self.actions[action][1]

        # Check if the agent reached a terminal state
        if self.current_state == self.terminal_state:
            is_terminal = True
            reward = 0.0
        else:
            # Check if the agent fell out of the boundaries of the grid world
            y_coord = self.current_state[0]
            x_coord = self.current_state[1]

            if y_coord >= self.num_rows:  # Agent went too far up
                self.current_state[0] = self.num_rows - 1
            elif y_coord < 0:  # Agent went too far down
                self.current_state[0] = 0

            if x_coord >= self.num_cols:  # Agent went too far right
                self.current_state[1] = self.num_cols - 1
            elif x_coord < 0:  # Agent went too far left
                self.current_state[1] = 0

            is_terminal = False
            reward = -1.0

        return reward, self.current_state, is_terminal
