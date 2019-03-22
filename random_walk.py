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
SIGMA_FACTOR = 0.95

NUM_EPISODES = 250
NUM_RUNS = 10


def rl_fixed_episode(agent, environment):
    # Get sigma value
    sigma = agent.sigma
    t = 0  # Time step
    tau = t - N + 1  # Time step to be updated
    terminal_t = np.inf  # Terminal time step

    # Initialize state, action, reward, and delta storage for n-step updates
    n_state = []
    n_action = []
    n_rwd = []
    n_delta = []

    # Initialize environment and agent
    start_state = environment.env_start()  # S_0
    start_action = agent.agent_start(start_state)  # A_0

    # Store starting state and action
    n_state.append(start_state)
    n_action.append(start_action)

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

        if (tau >= 0):
            # Get state, action, reward, and delta for the state-action pair to be updated
            upd_state = n_state[tau]
            upd_action = n_action[tau]

            e = 1
            g = q[upd_state][upd_action]

            for k in range(tau, min(tau+N-1, terminal_t-1) + 1):
                g += e*n_delta[k]
                e *= GAMMA*((1-sigma)*agent.prob_left + sigma)
            q[upd_state][upd_action] = q[upd_state][upd_action] + ALPHA*(g - q[upd_state][upd_action])
            agent.q = q
        t += 1
    # Episode complete
    agent.agent_end()


def rl_dynamic_episode(agent, environment):
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

            e = 1
            g = q[upd_state][upd_action]

            for k in range(tau, min(tau+N-1, terminal_t-1) + 1):
                g += e*n_delta[k]
                e *= GAMMA*((1-n_sigma[k])*agent.prob_left + n_sigma[k+1])
            q[upd_state][upd_action] = q[upd_state][upd_action] + ALPHA*(g - q[upd_state][upd_action])
            agent.q = q
        t += 1
    # Episode complete
    agent.agent_end()

def main():
    np.random.seed(10)
    matplotlib.rcParams.update({'font.size': 30})

    # Load analytic solution to compare with agent's q
    analytic_soln = np.load('analytic_soln.npy')

    episodes = range(1, NUM_EPISODES + 1)

    #== CONSTANT SIGMA ==#
    # Range of sigma values to iterate over
    sigma = [0, 0.25, 0.5, 0.75, 1]
    rms_err = np.full((NUM_EPISODES, NUM_RUNS, len(sigma)), 0, dtype=float)  # Stores RMS error between analytic and agent q
    for i in range(len(sigma)):
        sigma_val = sigma[i]
        for run in range(NUM_RUNS):
            fixed_agent = FixedRandomAgent(N, ALPHA, GAMMA, sigma_val)
            environment = WalkEnvironment()
            fixed_agent.agent_init()
            for j in range(NUM_EPISODES):
                rl_fixed_episode(fixed_agent, environment)
                ep_q = fixed_agent.q
                squared_err = np.power((np.subtract(analytic_soln, ep_q)), 2)
                tse = np.sum(squared_err)
                rms_err[j][run][i] = (tse / ep_q.size) ** (1/2)

    #== DYNAMIC SIGMA (EPISODE) ==#
    rms_err_ep = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)
    for run in range(NUM_RUNS):
        episodic_agent = EpisodicRandomAgent(N, ALPHA, GAMMA, SIGMA_FACTOR)
        environment = WalkEnvironment()
        episodic_agent.agent_init()
        for j in range(NUM_EPISODES):
            rl_fixed_episode(episodic_agent, environment)
            ep_q = episodic_agent.q
            squared_err = np.power((np.subtract(analytic_soln, ep_q)), 2)
            tse = np.sum(squared_err)
            rms_err_ep[j][run] = (tse / ep_q.size) ** (1 / 2)

    #== DYNAMIC SIGMA (FREQUENCY) ==#
    rms_err_freq = np.full((NUM_EPISODES, NUM_RUNS), 0, dtype=float)
    for run in range(NUM_RUNS):
        frequency_agent = FrequencyRandomAgent(N, ALPHA, GAMMA, SIGMA_FACTOR)
        environment = WalkEnvironment()
        frequency_agent.agent_init()
        for j in range(NUM_EPISODES):
            rl_dynamic_episode(frequency_agent, environment)
            ep_q = frequency_agent.q
            squared_err = np.power((np.subtract(analytic_soln, ep_q)), 2)
            tse = np.sum(squared_err)
            rms_err_freq[j][run] = (tse / ep_q.size) ** (1 / 2)

    # Plot final results
    mean_rms = np.mean(rms_err, axis=1)
    mean_rms_ep = np.mean(rms_err_ep, axis=1)
    mean_rms_freq = np.mean(rms_err_freq, axis=1)
    #for k in range(np.size(mean_rms, 1)):
    #    sigma_val = sigma[k]
    #    plt.plot(episodes, mean_rms[:, k], label=(r'$\sigma = {}$'.format(sigma_val)))
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
