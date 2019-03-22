import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fixed_agent import FixedRandomAgent
from episodic_dynamic_agent import EpisodicRandomAgent
from frequency_dynamic_agent import FrequencyRandomAgent
from walk_environment import WalkEnvironment

# CONSTANTS
NUM_STATES = 19
NUM_ACTIONS = 2

N = 3
ALPHA = 0.4
GAMMA = 1.0
SIGMA = [0, 0.25, 0.5, 0.75, 1]
SIGMA_FACTOR = 0.95

NUM_EPISODES = 250
NUM_RUNS = 10


def rl_episode(agent, environment):
    """
    Reinforcement learning episode for a Q(sigma) agent with fixed or dynamic episodic sigma
    :param agent: Fixed sigma agent
    :param environment: Environment the agent interacts with
    """
    # Get sigma value
    sigma = agent.sigma
    t = 0  # Time step
    tau = t - N + 1  # Time step to be updated
    terminal_t = np.inf  # Terminal time step - set as infinity for now until a terminal state is reached

    # Initialize state, action, reward, and delta storage for n-step updates
    n_state = []
    n_action = []
    n_rwd = []
    n_delta = []  # Error used to update Q

    # Initialize environment and agent
    start_state = environment.env_start()  # S_0
    start_action = agent.agent_start(start_state)  # A_0

    # Store starting state and action
    n_state.append(start_state)
    n_action.append(start_action)

    # Continue taking actions until a terminal state is reached
    while (t < terminal_t + N - 1):
        # Get current q function
        q = agent.q

        # Let the agent interact with the environment until a terminal state is reached
        if (t < terminal_t):
            # Take action A_t to get R_(t+1) and S_(t+1)
            (reward, next_state, is_terminal) = environment.env_step(n_action[len(n_action)-1])

            # Store reward
            n_rwd.append(reward)  # SAR for one time step

            # Get current state and action
            curr_state = n_state[t]
            curr_action = n_action[t]

            if is_terminal:  # Terminal state reached
                terminal_t = t + 1  # Update terminal time
                delta = reward - q[curr_state][curr_action]
            else:
                # Get action A_(t+1)
                next_action = agent.agent_step(next_state)

                # Calculate V_(t+1)
                V = 0.0
                for i in range(NUM_ACTIONS):
                    V += agent.prob_left*q[next_state][i]

                delta = reward + GAMMA*(sigma*q[next_state][next_action]+(1-sigma)*V) - q[curr_state][curr_action]

            # Add new state-action pair and delta to lists
            n_delta.append(delta)  # Time step t
            n_state.append(next_state)  # Time step t+1
            n_action.append(next_action)  # Time step t+1
        tau = t - N + 1

        # Perform n-step update (have enough steps to start updating Q)
        if (tau >= 0):
            # Get state, action, reward, and delta for the state-action pair to be updated
            upd_state = n_state[tau]
            upd_action = n_action[tau]

            # Initialize values used in n-step update
            e = 1
            g = q[upd_state][upd_action]

            # Update using the appropriate number of steps
            for k in range(tau, min(tau+N-1, terminal_t-1) + 1):
                g += e*n_delta[k]
                e *= GAMMA*((1-sigma)*agent.prob_left + sigma)
            q[upd_state][upd_action] = q[upd_state][upd_action] + ALPHA*(g - q[upd_state][upd_action])
            agent.q = q  # Update agent's Q
        t += 1

    # Episode complete
    agent.agent_end()


def rl_freq_episode(agent, environment):
    """
    Reinforcement learning episode for a Q(sigma) agent with dynamic frequency sigma
    :param agent: Dynamic sigma agent
    :param environment: Environment the agent interacts with
    """
    t = 0  # Time step
    tau = t - N + 1  # Time step to be updated
    terminal_t = np.inf  # Terminal time step

    # Initialize state, action, sigma, reward, and delta storage for n-step updates
    n_state = []
    n_action = []
    n_sigma = []
    n_rwd = []
    n_delta = []

    # Initialize environment and agent
    start_state = environment.env_start()  # S_0
    (start_action, start_sigma) = agent.agent_start(start_state)  # A_0

    # Store starting state and action
    n_state.append(start_state)
    n_action.append(start_action)
    n_sigma.append(start_sigma)

    # Continue taking actions until a terminal state is reached
    while (t < terminal_t + N - 1):
        # Get current q function
        q = agent.q

        # Let the agent interact with the environment until a terminal state is reached
        if (t < terminal_t):
            # Take action A_t to get R_(t+1) and S_(t+1)
            (reward, next_state, is_terminal) = environment.env_step(n_action[len(n_action)-1])

            # Store reward
            n_rwd.append(reward)  # SAR for one time step

            # Get current state and action
            curr_state = n_state[t]
            curr_action = n_action[t]

            if is_terminal:
                terminal_t = t + 1
                delta = reward - q[curr_state][curr_action]
            else:
                # Get action A_(t+1)
                (next_action, next_sigma) = agent.agent_step(next_state)

                # Calculate V_(t+1)
                V = 0.0
                for i in range(NUM_ACTIONS):
                    V += agent.prob_left*q[next_state][i]

                delta = reward + GAMMA*(next_sigma*q[next_state][next_action]+(1-next_sigma)*V) - q[curr_state][curr_action]

            # Add new state-action pair and delta to lists
            n_delta.append(delta)  # Time step t
            n_state.append(next_state)  # Time step t+1
            n_action.append(next_action)  # Time step t+1
            n_sigma.append(next_sigma)  # Time step t+1
        tau = t - N + 1

        if (tau >= 0):
            # Get state, action, reward, and delta for the state-action pair to be updated
            upd_state = n_state[tau]
            upd_action = n_action[tau]

            # Initialize values used in n-step update
            e = 1
            g = q[upd_state][upd_action]

            # Perform n-step update (have enough steps to start updating Q)
            for k in range(tau, min(tau+N-1, terminal_t-1) + 1):
                g += e*n_delta[k]
                e *= GAMMA*((1-n_sigma[k])*agent.prob_left + n_sigma[k+1])
            q[upd_state][upd_action] = q[upd_state][upd_action] + ALPHA*(g - q[upd_state][upd_action])
            agent.q = q  # Update agent's Q
        t += 1

    # Episode complete
    agent.agent_end()


def calc_ep_rmse(analytic_soln, q):
    """
    :param analytic_soln: Analytic (true) solution of the action-value function
    :param q: Action-value function found through RL
    :return: Root mean squared error
    """
    squared_err = np.power((np.subtract(analytic_soln, q)), 2)
    tse = np.sum(squared_err)
    rmse = (tse / q.size) ** (1 / 2)
    return rmse

def main():
    np.random.seed(10)  # Set random seed for consistency

    # Set plot configuration and variables
    matplotlib.rcParams.update({'font.size': 30})
    episodes = range(1, NUM_EPISODES + 1)  # x-axis
    
    # Load analytic solution to compare with agent's q
    analytic_soln = np.load('analytic_soln.npy')

    #== CONSTANT SIGMA ==#
    rms_err = np.full((NUM_EPISODES, NUM_RUNS, len(SIGMA)), 0, dtype=float)  # Stores RMS error between analytic and agent Q
    for i in range(len(SIGMA)):
        sigma_val = SIGMA[i]

        # Perform experiment for NUM_RUNS times
        for run in range(NUM_RUNS):
            fixed_agent = FixedRandomAgent(N, ALPHA, GAMMA, sigma_val)
            environment = WalkEnvironment()
            fixed_agent.agent_init()
            
            # Experiment consists of NUM_EPISODES episodes
            for j in range(NUM_EPISODES):
                rl_episode(fixed_agent, environment)
                ep_q = fixed_agent.q

                # Determine RMSE for the current episode number
                rms_err[j][run][i] = calc_ep_rmse(analytic_soln, ep_q)

    # Average results over the total number of experiments (runs)
    mean_rms = np.mean(rms_err, axis=1)

    #== DYNAMIC SIGMA (EPISODE) ==#
    rms_err_ep = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)

    # Perform experiment for NUM_RUNS times
    for run in range(NUM_RUNS):
        episodic_agent = EpisodicRandomAgent(N, ALPHA, GAMMA, SIGMA_FACTOR)
        environment = WalkEnvironment()
        episodic_agent.agent_init()

        # Experiment consists of NUM_EPISODES episodes
        for j in range(NUM_EPISODES):
            rl_episode(episodic_agent, environment)
            ep_q = episodic_agent.q

            # Determine RMSE for the current episode number
            rms_err_ep[j][run] = calc_ep_rmse(analytic_soln, ep_q)

    # Average results over the total number of experiments (runs)
    mean_rms_ep = np.mean(rms_err_ep, axis=1)

    #== DYNAMIC SIGMA (FREQUENCY) ==#
    rms_err_freq = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)

    # Perform experiment for NUM_RUNS times
    for run in range(NUM_RUNS):
        frequency_agent = FrequencyRandomAgent(N, ALPHA, GAMMA, SIGMA_FACTOR)
        environment = WalkEnvironment()
        frequency_agent.agent_init()

        # Experiment consists of NUM_EPISODES episodes
        for j in range(NUM_EPISODES):
            rl_freq_episode(frequency_agent, environment)
            ep_q = frequency_agent.q

            # Determine RMSE for the current episode number
            rms_err_freq[j][run] = calc_ep_rmse(analytic_soln, ep_q)

    # Average results over the total number of experiments (runs)
    mean_rms_freq = np.mean(rms_err_freq, axis=1)

    # Plot final results
    for k in range(np.size(mean_rms, 1)):
        sigma_val = SIGMA[k]
        plt.plot(episodes, mean_rms[:, k], label=(r'$\sigma = {}$'.format(sigma_val)))
    plt.plot(episodes, mean_rms_ep, label=r'Dynamic $\sigma$ (Episode)')
    plt.plot(episodes, mean_rms_freq, label=r'Dynamic $\sigma$ (Frequency)')
    plt.xlabel('Episodes')
    plt.ylabel('RMS Error')
    plt.title(r'Q($\sigma$) Curves for 19-State Random Walk')
    plt.legend()
    plt.show()

    print("Complete\n")


if __name__ == '__main__':
    main()
