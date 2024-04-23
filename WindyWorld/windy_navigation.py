import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

from episodic_agent import EpisodicAgent
from frequency_raw_agent import FrequencyRawAgent
from frequency_mean_agent import FrequencyMeanAgent
from windy_environment import WindyEnvironment
from multiprocessing import Pool, freeze_support, cpu_count

# CONSTANTS
# Environment
NUM_ACTIONS = 4

# Agent
N = 1
ALPHA = np.array([1/16, 1/8, 1/4, 1/2, 3/4, 1])
GAMMA = 1.0
EPSILON = 0.1
SIGMA = np.array([0, 0.5, 1])
EP_SIGMA_FACTOR = 0.95
FREQ_SIGMA_FACTOR = 0.9

# Experiment
NUM_EPISODES = 100
NUM_RUNS = 1000
MAX_STEPS = 1000  # Stopping criterion


def rl_episode(agent, environment):
    """
    Reinforcement learning episode for a Q(sigma) agent with fixed or dynamic episodic sigma
    :param agent: Q(sigma) agent
    :param environment: Environment the agent interacts with
    :return: Total reward
    """
    t = 0  # Time step
    tau = t - N + 1  # Time step to be updated
    terminal_t = np.inf  # Terminal time step - set as infinity for now until a terminal state is reached

    # Initialize state, action, reward, sigma, delta, and probability storage for n-step updates
    n_state = []
    n_action = []
    n_rwd = []
    n_sigma = []
    n_delta = []  # Error used to update Q
    n_prob = []  # Probability of the agent choosing the action it took

    # Initialize environment and agent
    start_state = environment.env_start()  # S_0
    (start_action, start_pi, start_sigma) = agent.agent_start(start_state)  # A_0

    # Store starting state, action, sigma, and probability
    n_state.append(start_state[:])
    n_action.append(start_action)
    n_sigma.append(start_sigma)
    n_prob.append(start_pi[start_action])

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
                delta = reward - q[curr_state[0]][curr_state[1]][curr_action]
            else:
                # Get action A_(t+1)
                (next_action, next_pi, next_sigma) = agent.agent_step(next_state)

                # Calculate TD error
                exp_target = np.dot(next_pi, q[next_state[0]][next_state[1]][:])  # V_(t+1)
                sarsa_target = q[next_state[0]][next_state[1]][next_action]
                delta = reward + agent.gamma*(next_sigma*sarsa_target+(1-next_sigma)*exp_target) - q[curr_state[0]][curr_state[1]][curr_action]

            # Add new state-action pair and delta to lists
            n_delta.append(delta)  # Time step t
            n_state.append(next_state[:])  # Time step t+1
            n_action.append(next_action)  # Time step t+1
            n_prob.append(next_pi[next_action])  # Time step t+1
            n_sigma.append(next_sigma)  # Time step t+1
        tau = t - N + 1

        # Perform n-step update (have enough steps to start updating Q)
        if (tau >= 0):
            # Get state, action, reward, and delta for the state-action pair to be updated
            upd_state = n_state[tau]
            upd_action = n_action[tau]

            # Initialize values used in n-step update
            e = 1
            g = q[upd_state[0]][upd_state[1]][upd_action]

            # Update using the appropriate number of steps
            for k in range(tau, min(tau+N-1, terminal_t-1) + 1):
                g += e*n_delta[k]
                e *= agent.gamma*((1-n_sigma[k])*n_prob[k+1] + n_sigma[k+1])
            q[upd_state[0]][upd_state[1]][upd_action] += agent.alpha*(g - q[upd_state[0]][upd_state[1]][upd_action])
            agent.q = np.copy(q)  # Update agent's Q
        t += 1
        if (t > MAX_STEPS):
            break  # Avoid divergence by stopping early

    # Episode complete
    agent.agent_end()

    return np.sum(n_rwd)


def rl_run(agent, environment, num_episodes):
    """
    Performs one run of an agent with the environment for a given number of episodes
    :param agent: Q(sigma) agent
    :param environment: Environment that the agent interacts with
    :param num_episodes: Total number of episodes the agent completes
    :return: Average return per episode
    """
    tot_rwd = np.zeros(num_episodes)  # Initialize array that stores the total reward for each episode
    agent.agent_reset()  # Reset agent

    # Run the agent through the environment for num_episodes
    for i in range(num_episodes):
        tot_rwd[i] = rl_episode(agent, environment)
    mean_rwd = np.mean(tot_rwd)

    return mean_rwd


def rl_experiment(agent_class, n, alpha, gamma, sigma, sigma_factor, epsilon, env_class, num_episodes, num_runs):
    """
    Determines the root mean squared error between the agent's q after a number of episodes and the true analytical solution, averaged over a number of runs
    :param agent_class: Q(sigma) agent class
    :param n: Number of steps used in update
    :param alpha: Step size
    :param gamma: Discount factor
    :param sigma: Starting degree of sampling
    :param sigma_factor: Multiplicative factor used to reduce sigma
    :param epsilon: Probability of agent taking a random action
    :param env_class: Environment class that the agent interacts with
    :param num_episodes: Total number of episodes the agent completes
    :param num_runs: Total number of runs the agent completes
    :return: Mean average return per episode, standard error of average return per episode
    """
    agent = agent_class(n, alpha, gamma, sigma, sigma_factor, epsilon)
    environment = env_class()
    # Parallelize over the number of runs
    with Pool(cpu_count()) as pool:
       mean_rwd = pool.starmap(rl_run, itertools.repeat((agent, environment, num_episodes), times=num_runs))
    mean_mean_rwd = np.mean(mean_rwd, axis=0)
    stde_mean_rwd = np.std(mean_rwd, axis=0) / np.sqrt(num_runs)

    return mean_mean_rwd, stde_mean_rwd


def main():
    np.random.seed(10)  # Set random seed for consistency

    # Set plot configuration and variables
    matplotlib.rcParams.update({'font.size': 30})

    #== CONSTANT SIGMA ==#
    # mean_rwd = np.zeros((len(ALPHA), len(SIGMA)))
    # stde_rwd = np.zeros((len(ALPHA), len(SIGMA)))
    # for i in range(len(SIGMA)):
    #     sigma_val = SIGMA[i]
    #
    #     for j in range(len(ALPHA)):
    #         alpha_val = ALPHA[j]
    #         mean_rwd[j, i], stde_rwd[j, i] = rl_experiment(EpisodicAgent, N, alpha_val, GAMMA, sigma_val, 1, EPSILON, WindyEnvironment, NUM_EPISODES, NUM_RUNS)

    #== DYNAMIC SIGMA (EPISODE) ==#
    mean_rwd_ep = np.zeros(len(ALPHA))
    stde_rwd_ep = np.zeros(len(ALPHA))
    for j in range(len(ALPHA)):
        alpha_val = ALPHA[j]
        mean_rwd_ep[j], stde_rwd_ep[j] = rl_experiment(EpisodicAgent, N, alpha_val, GAMMA, 1, EP_SIGMA_FACTOR, EPSILON, WindyEnvironment, NUM_EPISODES, NUM_RUNS)

    #== DYNAMIC SIGMA (FREQUENCY, RAW) ==#
    mean_rwd_freq_raw = np.zeros(len(ALPHA))
    stde_rwd_freq_raw = np.zeros(len(ALPHA))
    for j in range(len(ALPHA)):
        alpha_val = ALPHA[j]
        mean_rwd_freq_raw[j], stde_rwd_freq_raw[j] = rl_experiment(FrequencyRawAgent, N, alpha_val, GAMMA, 1, FREQ_SIGMA_FACTOR, EPSILON, WindyEnvironment, NUM_EPISODES, NUM_RUNS)

    #== DYNAMIC SIGMA (FREQUENCY, MEAN) ==#
    mean_rwd_freq_mean = np.zeros(len(ALPHA))
    stde_rwd_freq_mean = np.zeros(len(ALPHA))
    for j in range(len(ALPHA)):
        alpha_val = ALPHA[j]
        mean_rwd_freq_mean[j], stde_rwd_freq_mean[j] = rl_experiment(FrequencyMeanAgent, N, alpha_val, GAMMA, 1, FREQ_SIGMA_FACTOR, EPSILON, WindyEnvironment, NUM_EPISODES, NUM_RUNS)

    n1_mean = np.vstack((mean_rwd_ep, mean_rwd_freq_raw, mean_rwd_freq_mean))
    n1_stde = np.vstack((stde_rwd_ep, stde_rwd_freq_raw, stde_rwd_freq_mean))
    n1_results = np.stack((n1_mean, n1_stde), axis=-1)
    np.save('C:/Users/Jason/Dropbox/University/Grad School/Winter Term/CMPUT 609/Project/n1_results.npy', n1_results)

    # Plot final results
    plt.figure(figsize=(18.5, 10.5))
    plt.plot(ALPHA, mean_rwd_ep, label=r'Dynamic $\sigma$ (Episode)')
    plt.plot(ALPHA, mean_rwd_freq_raw, label=r'Dynamic $\sigma$ (Frequency, Raw)')
    plt.plot(ALPHA, mean_rwd_freq_mean, label=r'Dynamic $\sigma$ (Frequency, Mean)')
    # for k in range(len(SIGMA)):
    #     sigma_val = SIGMA[k]
    #     plt.plot(ALPHA, mean_rwd[:, k], label=(r'$\sigma= {}$'.format(sigma_val)))
    plt.xlabel(r'Step Size $\alpha$')
    plt.ylabel('Average Return per Episode')
    plt.title(r'Q($\sigma$) Curves for Stochastic Windy Grid World')
    plt.legend(prop={'size': 20})
    plt.savefig('C:/Users/Jason/Dropbox/University/Grad School/Winter Term/CMPUT 609/Project/windy_results.svg')

    # Plot with error bars
    plt.figure(figsize=(18.5, 10.5))
    plt.errorbar(ALPHA, mean_rwd_ep, yerr=stde_rwd_ep, capsize=5, label=r'Dynamic $\sigma$ (Episode)')
    plt.errorbar(ALPHA, mean_rwd_freq_raw, yerr=stde_rwd_freq_raw, capsize=5, label=r'Dynamic $\sigma$ (Frequency, Raw)')
    plt.errorbar(ALPHA, mean_rwd_freq_mean, yerr=stde_rwd_freq_mean, capsize=5, label=r'Dynamic $\sigma$ (Frequency, Mean)')
    # for k in range(len(SIGMA)):
    #     sigma_val = SIGMA[k]
    #     plt.errorbar(ALPHA, mean_rwd[:, k], yerr=stde_rwd[:, k], capsize=5, label=(r'$\sigma = {}$'.format(sigma_val)))
    plt.xlabel(r'Step Size $\alpha$')
    plt.ylabel('Average Return per Episode')
    plt.title(r'Q($\sigma$) Curves for Stochastic Windy Grid World with Error Bars')
    plt.legend(prop={'size': 20})
    plt.savefig('C:/Users/Jason/Dropbox/University/Grad School/Winter Term/CMPUT 609/Project/windy_results_errbar.svg')
    plt.show()

    print("Complete\n")


if __name__ == '__main__':
    freeze_support()
    main()
