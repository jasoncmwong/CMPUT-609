import numpy as np

NUM_STATES = 21
NUM_ACTIONS = 2
NUM_ITER = 1e6
GAMMA = 1

analytic_soln = np.zeros((NUM_STATES, NUM_ACTIONS))
env_prob = np.full((NUM_STATES, NUM_ACTIONS), 0.5, dtype=float)

# Set probabilities of terminal states to 0, indicating that the episode ends
env_prob[0, :] = np.array([0, 0])
env_prob[NUM_STATES - 1, :] = np.array([0, 0])

env_reward = np.zeros(NUM_STATES)
env_reward[0] = -1  # Leftmost state
env_reward[NUM_STATES - 1] = 1  # Rightmost state

actions = np.array([-1, 1])

###
# Deterministic: p(s',r | s,a) = 1 for all (s,a)
# Undiscounted: gamma = 1
###


iter = 0
while (iter < NUM_ITER):
    for i in range(1, NUM_STATES-1):  # Ignore terminal states
        for j in range(NUM_ACTIONS):
            action = actions[j]
            next_state = i + action
            expectation = 0
            for k in range(NUM_ACTIONS):
                expectation = expectation + env_prob[next_state][k]*analytic_soln[next_state][k]
            analytic_soln[i][j] = env_reward[next_state] + GAMMA*expectation
    iter = iter + 1

# Truncate value function back to 19 states
analytic_soln = np.delete(analytic_soln, -1, axis=0)
analytic_soln = np.delete(analytic_soln, 0, axis=0)
np.save('analytic_soln', analytic_soln)
print("Complete\n")
