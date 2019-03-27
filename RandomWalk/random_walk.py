import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial

from episodic_dynamic_agent import EpisodicRandomAgent
from frequency_dynamic_agent import FrequencyRandomAgent
from walk_environment import WalkEnvironment

# CONSTANTS
NUM_STATES = 19
NUM_ACTIONS = 2

N = 3
ALPHA = 0.4
GAMMA = 1.0
SIGMA = np.array([0, 0.25, 0.5, 0.75, 1])
SIGMA_FACTOR = 0.95

NUM_EPISODES = 250
NUM_RUNS = 10

# Load analytic solution to compare with agent's q
ANALYTIC_SOLN = np.load('analytic_soln.npy')


def rl_episode(agent, environment):
    """
    Reinforcement learning episode for a Q(sigma) agent with fixed or dynamic episodic sigma
    :param agent: Q(sigma) agent
    :param environment: Environment the agent interacts with
    """
    t = 0  # Time step
    tau = t - N + 1  # Time step to be updated
    terminal_t = np.inf  # Terminal time step - set as infinity for now until a terminal state is reached

    # Initialize state, action, reward, and delta storage for n-step updates
    n_state = []
    n_action = []
    n_sigma = []
    n_rwd = []
    n_delta = []  # Error used to update Q

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
        q = np.copy(agent.q)

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
                (next_action, next_sigma) = agent.agent_step(next_state)

                # Calculate TD error
                exp_target = np.sum(agent.prob_left*q[next_state, :])  # V_(t+1)
                sarsa_target = q[next_state][next_action]
                delta = reward + agent.gamma*(next_sigma*sarsa_target+(1-next_sigma)*exp_target) - q[curr_state][curr_action]

            # Add new state-action pair and delta to lists
            n_delta.append(delta)  # Time step t
            n_state.append(next_state)  # Time step t+1
            n_action.append(next_action)  # Time step t+1
            n_sigma.append(next_sigma)  # Time step t+1
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
                e *= agent.gamma*((1-n_sigma[k])*agent.prob_left + n_sigma[k+1])
            q[upd_state][upd_action] = q[upd_state][upd_action] + agent.alpha*(g - q[upd_state][upd_action])
            agent.q = np.copy(q)  # Update agent's Q
        t += 1

    # Episode complete
    agent.agent_end()


def rl_experiment(agent, n, alpha, gamma, sigma, sigma_factor, environment, num_episodes, num_runs):
    """
    Reinforcement learning experiment that determines the root mean squared error of the agent's Q after each episode
    :param agent: Q(sigma) agent
    :param n: Number of steps used in update
    :param alpha: Step size
    :param gamma: Discount factor
    :param sigma: Starting degree of sampling
    :param sigma_factor: Multiplicative factor used to reduce sigma
    :param environment: Environment the agent interacts with
    :param num_episodes: Total number of episodes the agent completes
    :param num_runs: Total number of runs the agent completes
    :return: Root mean squared error of Q for each episode
    """
    rmse = np.zeros(num_episodes, num_runs)  # Initialize array that stores RMSE for each episode

    agent.agent_init(n, alpha, gamma, sigma, sigma_factor)  # Initialize agent

    for i in range(num_runs):
        # Run the agent through the environment for num_episodes
        for j in range(num_episodes):
            rl_episode(agent, environment)
            ep_q = agent.q
            rmse[i][j] = calc_ep_rmse(ANALYTIC_SOLN, ep_q)  # Calculate RMSE for current episode number

    return rmse


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

    #== CONSTANT SIGMA ==#
    #rms_err = np.full((NUM_EPISODES, NUM_RUNS, len(SIGMA)), 0, dtype=float)  # Stores RMS error between analytic and agent Q

    with Pool(cpu_count()) as pool:
        rms_err = pool.map(partial(rl_experiment,
                                   agent=EpisodicRandomAgent(),
                                   n=N,
                                   alpha=ALPHA,
                                   gamma=GAMMA,
                                   sigma_factor=SIGMA_FACTOR,
                                   environment=WalkEnvironment(),
                                   num_episodes=NUM_EPISODES,
                                   num_runs=NUM_RUNS),
                           SIGMA)

    # Average results over the total number of experiments (runs)
    mean_rms = np.mean(rms_err, axis=1)

    #== DYNAMIC SIGMA (EPISODE) ==#
    # Create agent and environment objects
    episodic_agent = EpisodicRandomAgent()
    environment = WalkEnvironment()

    rms_err_ep = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)
    # Perform experiment for NUM_RUNS times
    for run in range(NUM_RUNS):
        rms_err_ep[:, run] = rl_experiment(episodic_agent, N, ALPHA, GAMMA, 1, SIGMA_FACTOR, environment, NUM_EPISODES, NUM_RUNS)

    # Average results over the total number of experiments (runs)
    mean_rms_ep = np.mean(rms_err_ep, axis=1)

    #== DYNAMIC SIGMA (FREQUENCY, RAW) ==#
    # Create agent and environment objects
    freq_raw_agent = FrequencyRandomAgent(use_mean=False)
    environment = WalkEnvironment()

    rms_err_freq_raw = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)
    # Perform experiment for NUM_RUNS times
    for run in range(NUM_RUNS):
        rms_err_freq_raw[:, run] = rl_experiment(freq_raw_agent, N, ALPHA, GAMMA, 1, SIGMA_FACTOR, environment, NUM_EPISODES, NUM_RUNS)

    # Average results over the total number of experiments (runs)
    mean_rms_freq_raw = np.mean(rms_err_freq_raw, axis=1)

    #== DYNAMIC SIGMA (FREQUENCY, MEAN) ==#
    # Create agent and environment objects
    freq_mean_agent = FrequencyRandomAgent(use_mean=True)
    environment = WalkEnvironment()

    rms_err_freq_mean = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)
    # Perform experiment for NUM_RUNS times
    for run in range(NUM_RUNS):
        rms_err_freq_mean[:, run] = rl_experiment(freq_mean_agent, N, ALPHA, GAMMA, 1, SIGMA_FACTOR, environment, NUM_EPISODES)

    # Average results over the total number of experiments (runs)
    mean_rms_freq_mean = np.mean(rms_err_freq_mean, axis=1)

    # Plot final results
    for k in range(np.size(mean_rms, 1)):
        sigma_val = SIGMA[k]
        plt.plot(episodes, mean_rms[:, k], label=(r'$\sigma = {}$'.format(sigma_val)))
    plt.plot(episodes, mean_rms_ep, label=r'Dynamic $\sigma$ (Episode)')
    plt.plot(episodes, mean_rms_freq_raw, label=r'Dynamic $\sigma$ (Frequency, Raw)')
    plt.plot(episodes, mean_rms_freq_mean, label=r'Dynamic $\sigma$ (Frequency, Mean)')
    plt.xlabel('Episodes')
    plt.ylabel('RMS Error')
    plt.title(r'Q($\sigma$) Curves for 19-State Random Walk')
    plt.legend()
    plt.show()

    print("Complete\n")


if __name__ == '__main__':
    main()
