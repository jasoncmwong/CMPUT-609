import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial

from episodic_agent import EpisodicRandomAgent
from frequency_raw_agent import FrequencyRawRandomAgent
from frequency_mean_agent import FrequencyMeanRandomAgent
from walk_environment import WalkEnvironment

#== CONSTANTS ==#
# Environment
NUM_STATES = 19
NUM_ACTIONS = 2

# Agent
N = 3
ALPHA = 0.4
GAMMA = 1.0
SIGMA = np.array([0, 0.25, 0.5, 0.75, 1])
SIGMA_FACTOR = 0.95

# Experiment
NUM_EPISODES = 100
NUM_RUNS = 10
ANALYTIC_SOLN = np.load('analytic_soln.npy')  # Analytical solution


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


def rl_experiment(sigma, agent_class, n, alpha, gamma, sigma_factor, env_class, num_episodes, num_runs):
    """
    Determines the root mean squared error between the agent's q after a number of episodes and the true analytical solution, averaged over a number of runs
    :param agent_class: Q(sigma) agent class
    :param n: Number of steps used in update
    :param alpha: Step size
    :param gamma: Discount factor
    :param sigma: Starting degree of sampling
    :param sigma_factor: Multiplicative factor used to reduce sigma
    :param env_class: Environment class that the agent interacts with
    :param num_episodes: Total number of episodes the agent completes
    :param num_runs: Total number of runs the agent completes
    :return: Root mean squared error of Q for each episode, standard error of Q for each episode
    """
    rmse = np.zeros((num_episodes, num_runs))  # Initialize array that stores RMSE for each episode
    agent = agent_class(n, alpha, gamma, sigma, sigma_factor)
    environment = env_class()

    for i in range(num_runs):
        agent.agent_reset()  # Reset agent by re-initializing its action-value function
        # Run the agent through the environment for num_episodes
        for j in range(num_episodes):
            rl_episode(agent, environment)
            ep_q = agent.q
            squared_err = np.power((np.subtract(ANALYTIC_SOLN, ep_q)), 2)
            tse = np.sum(squared_err)
            rmse[j][i] = (tse / ep_q.size) ** (1 / 2)  # Calculate RMSE for current episode number
    mean_rmse = np.mean(rmse, axis=1)
    stde_rmse = np.std(rmse, axis=1) / np.sqrt(num_runs)
    return mean_rmse, stde_rmse


def main():
    np.random.seed(10)  # Set random seed for consistency

    # Set plot configuration and variables
    matplotlib.rcParams.update({'font.size': 30})
    episodes = range(1, NUM_EPISODES + 1)  # x-axis

    #== CONSTANT SIGMA ==#
    # Parallelize experiments over the constant sigma values
    with Pool(cpu_count()) as pool:
        fixed_results = pool.map(partial(rl_experiment,
                                         agent_class=EpisodicRandomAgent,
                                         n=N,
                                         alpha=ALPHA,
                                         gamma=GAMMA,
                                         sigma_factor=1,
                                         env_class=WalkEnvironment,
                                         num_episodes=NUM_EPISODES,
                                         num_runs=NUM_RUNS),
                                 SIGMA)
    mean_rmse, stde_rmse = zip(*fixed_results)

    #== DYNAMIC SIGMA (EPISODE) ==#
    mean_rmse_ep, stde_rmse_ep = rl_experiment(1, EpisodicRandomAgent, N, ALPHA, GAMMA, SIGMA_FACTOR, WalkEnvironment, NUM_EPISODES, NUM_RUNS)

    #== DYNAMIC SIGMA (FREQUENCY, RAW) ==#
    mean_rmse_freq_raw, stde_rmse_freq_raw = rl_experiment(1, FrequencyRawRandomAgent, N, ALPHA, GAMMA, SIGMA_FACTOR, WalkEnvironment, NUM_EPISODES, NUM_RUNS)

    #== DYNAMIC SIGMA (FREQUENCY, MEAN) ==#
    mean_rmse_freq_mean, stde_rmse_freq_mean = rl_experiment(1, FrequencyMeanRandomAgent, N, ALPHA, GAMMA, SIGMA_FACTOR, WalkEnvironment, NUM_EPISODES, NUM_RUNS)

    # Plot final results
    plt.figure()
    plt.plot(episodes, mean_rmse_ep, label=r'Dynamic $\sigma$ (Episode)')
    plt.plot(episodes, mean_rmse_freq_raw, label=r'Dynamic $\sigma$ (Frequency, Raw)')
    plt.plot(episodes, mean_rmse_freq_mean, label=r'Dynamic $\sigma$ (Frequency, Mean)')
    for k in range(len(mean_rmse)):
        sigma_val = SIGMA[k]
        plt.plot(episodes, mean_rmse[k][:], label=(r'$\sigma = {}$'.format(sigma_val)))
    plt.xlabel('Episodes')
    plt.ylabel('RMS Error')
    plt.title(r'Q($\sigma$) Curves for 19-State Random Walk')
    plt.legend(prop={'size': 20})

    # Plot with errorbars
    plt.figure()
    plt.errorbar(episodes, mean_rmse_ep, yerr=stde_rmse_ep, capsize=5, label=r'Dynamic $\sigma$ (Episode)')
    plt.errorbar(episodes, mean_rmse_freq_raw, yerr=stde_rmse_freq_raw, capsize=5, label=r'Dynamic $\sigma$ (Frequency, Raw)')
    plt.errorbar(episodes, mean_rmse_freq_mean, yerr=stde_rmse_freq_mean, capsize=5, label=r'Dynamic $\sigma$ (Frequency, Mean)')
    # for k in range(len(mean_rmse)):
    #     sigma_val = SIGMA[k]
    #     plt.errorbar(episodes, mean_rmse[k][:], yerr=stde_rmse[k][:], capsize=5, label=(r'$\sigma = {}$'.format(sigma_val)))
    plt.xlabel('Episodes')
    plt.ylabel('RMS Error')
    plt.title(r'Dynamic Q($\sigma$) Curves for 19-State Random Walk with Error Bars')
    plt.legend(prop={'size': 20})

    plt.show()


    print("Complete\n")


if __name__ == '__main__':
    main()
