from collections import defaultdict

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from gym.envs.toy_text import FrozenLakeEnv


############################### env method : you don't need to know them


def modify_rewards(next_state, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    state_type = custom_map_flaten[next_state]

    if state_type == "H":
        return hole_reward  # Decrease the reward for falling into a hole
    elif state_type == "G":
        return goal_reward  # Increase the reward for reaching the goal
    else:
        return move_reward  # Decrease the reward for moving


def modify_rewards_P(envP, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    # old_envP = copy.deepcopy(envP)
    old_envP = copy.copy(envP)

    new_envP = {}
    for state, v1 in old_envP.items():
        inner_dict = {}
        for action, v2 in v1.items():
            inner_list = []
            for tpl in v2:
                (prob_of_transition, s_prime, old_reward, terminated) = tpl
                if custom_map_flaten[s_prime] == "H":
                    new_reward = (
                        hole_reward  # Decrease the reward for falling into a hole
                    )
                elif custom_map_flaten[s_prime] == "G":
                    new_reward = (
                        goal_reward  # Increase the reward for reaching the goal
                    )
                else:
                    new_reward = move_reward  # Decrease the reward for movin
                inner_list.append((prob_of_transition, s_prime, new_reward, terminated))
            inner_dict[action] = inner_list
        new_envP[state] = inner_dict

    return new_envP


class ModifyRewards(gym.Wrapper):
    def __init__(
            self, env, custom_map, hole_reward=-10, goal_reward=10, move_reward=-0.1
    ):
        super().__init__(env)
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.move_reward = move_reward
        self.custom_map = custom_map
        self.P = modify_rewards_P(
            env.P, custom_map, hole_reward, goal_reward, move_reward
        )

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        modified_reward = modify_rewards(
            next_state,
            self.custom_map,
            self.hole_reward,
            self.goal_reward,
            self.move_reward,
        )
        return next_state, modified_reward, done, truncated, info


############################### plot methods : you can use them to plot 
# your policy and state value


#  plot policy with arrows in four direction to understand policy better
def plot_policy_arrows(policy, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    n = len(custom_map)
    m = len(custom_map[0])
    fig, ax = plt.subplots(n, m, figsize=(8, 8))
    for i in range(n):
        for j in range(m):
            ax[i, j].set_xlim([0, 3])
            ax[i, j].set_ylim([0, 3])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    for state, subdict in policy.items():
        row = state // m
        col = state % m
        if custom_map_flaten[state] == "S":
            square_fill = plt.Rectangle(
                (0.5, 0.5), 2, 2, linewidth=0, edgecolor=None, facecolor="y", alpha=0.5
            )
            ax[row, col].add_patch(square_fill)
        for direction, value in subdict.items():
            dx, dy = 0, 0
            if direction == 0:
                dx = -value
            elif direction == 1:
                dy = -value
            elif direction == 2:
                dx = value
            else:
                dy = value
            if value != 0:
                ax[row, col].arrow(1.5, 1.5, dx, dy, head_width=0.35, head_length=0.25)
        if subdict[0] == 0 and subdict[1] == 0 and subdict[2] == 0 and subdict[3] == 0:
            if custom_map_flaten[state] == "G":
                color = "g"
            else:
                color = "r"
            square_fill = plt.Rectangle(
                (0.5, 0.5),
                2,
                2,
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=0.5,
            )
            ax[row, col].add_patch(square_fill)
    plt.show()


# plot policy in terminal using best action for each state
def plot_policy_terminal(policy, custom_map):
    arr = np.empty((len(custom_map), len(custom_map[0])), dtype=object)
    state = 0
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            subdict = policy[state]
            best_action = max(subdict, key=subdict.get)
            if best_action == 0:
                arr[i, j] = "Lt"  # 0: LEFT
            elif best_action == 1:
                arr[i, j] = "Dn"  # 1: DOWN
            elif best_action == 2:
                arr[i, j] = "Rt"  # 2: RIGHT
            elif best_action == 3:
                arr[i, j] = "UP"  # 3: UP
            else:
                arr[i, j] = "##"
            state += 1
    print("Policy Terminal: \n", arr)


# plot state value
def plot_state_value(state_value, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    rows = len(custom_map)
    cols = len(custom_map[0])
    table = state_value.reshape(rows, cols)
    # Define custom colors
    green = mcolors.to_rgba("green", alpha=0.5)
    blue = mcolors.to_rgba("blue", alpha=0.5)
    fig, ax = plt.subplots()
    im = ax.imshow(table, cmap="Reds")
    state = 0
    for i in range(rows):
        for j in range(cols):
            if custom_map_flaten[state] == "H":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0.1,
                        facecolor=blue,
                    )
                )
            elif custom_map_flaten[state] == "G":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0,
                        facecolor=green,
                    )
                )

            ax.text(
                j,
                i,
                str(i * cols + j) + "\n" + custom_map_flaten[state] + "\n" + "{:.2f}".format(table[i, j]),
                ha="center",
                va="center",
            )
            state += 1
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([""] * cols)
    ax.set_yticklabels([""] * rows)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
    ax.set_title("state value")
    plt.show()


############################### handler methods : you don't need to know them,
# they have been used in other methods


def act_wrt_prob(probability):
    if random.random() < probability:
        return 1
    else:
        return 0


def get_action_wrt_policy(state, policy):
    action = -1
    while action == -1:
        if act_wrt_prob(policy[state][0]) == 1:
            action = 0
        elif act_wrt_prob(policy[state][1]) == 1:
            action = 1
        elif act_wrt_prob(policy[state][2]) == 1:
            action = 2
        elif act_wrt_prob(policy[state][3]) == 1:
            action = 3
    return action


def get_flaten_custom_map(custom_map):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    return custom_map_flaten


############################### helper methods : you can use them in your code to create
# random policy and check your policy


# it gives a randome walk policy w.r.t costum 
def get_init_policy(custom_map):
    policy = {}
    random_sub_dict = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":

                policy[state] = terminal_sub_dict
            else:
                policy[state] = random_sub_dict

    return policy


# it gives walk policy according to direction w.r.t costum
def get_policy_direction(direction, custom_map):  # direction :"left", "down", "right"
    policy = {}
    left_sub_dict = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    down_sub_dict = {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0}
    right_sub_dict = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":

                policy[state] = terminal_sub_dict
            else:
                if direction == "left":
                    policy[state] = left_sub_dict
                elif direction == "down":
                    policy[state] = down_sub_dict
                elif direction == "right":
                    policy[state] = right_sub_dict

    return policy


# it run game according to given policy
def do_policy(env, policy, episdoes=5):
    for ep in range(episdoes):
        n_state = env.reset()[0]
        done = False
        rewards = 0
        moves = 0
        while done is False:
            action = get_action_wrt_policy(n_state, policy)
            n_state, reward, done, truncated, info = env.step(action)
            rewards += reward
            moves += 1
        print("rewards:", rewards, " - moves:", moves, " - final state:", n_state, " - ", format_policy(policy))
    env.render()


############################### algorithm methods : you have to implement these algorithms

def policy_iteration(env, custom_map, max_ittr=30, theta=0.01, discount_factor=0.9):
    """
    Perform policy iteration on a FrozenLake environment using the Policy Evaluation and Policy Improvement steps.

    Args:
    - env: FrozenLake environment to perform policy iteration on.
    - custom_map: List of strings specifying the layout of the FrozenLake environment.
    - max_ittr: Maximum number of policy iterations to perform (default: 30).
    - theta: Threshold for convergence of state values during policy evaluation (default: 0.01).
    - discount_factor: Discount factor for future rewards (default: 0.9).

    Returns:
    - V: A numpy array of state values.
    - policy: A dictionary containing the optimal policy for each state.
    """

    # Initialize the policy and state value function
    policy = get_init_policy(custom_map)
    V = np.zeros(env.observation_space.n)

    # Get the transition probabilities and rewards for each state-action pair
    P = env.P

    # Loop until the policy is stable or maximum number of iterations is reached
    ittr = 0
    policy_stable = False
    while not policy_stable and ittr < max_ittr:
        # Policy Evaluation
        while True:
            delta = 0
            # Update the value of each state using the Bellman equation
            for s in range(env.observation_space.n):
                v = V[s]
                # Compute the expected value of each action using the current policy
                action_values = [sum(p * (r + discount_factor * V[s1]) for (p, s1, r, _) in P[s][a]) for a in
                                 range(env.action_space.n)]
                # Update the state value with the maximum action value
                V[s] = max(action_values)
                delta = max(delta, abs(v - V[s]))
            # Check for convergence
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(env.observation_space.n):
            old_action = max(policy[s], key=policy[s].get)
            # Compute the expected value of each action using the updated state values
            action_values = [sum(p * (r + discount_factor * V[s1]) for (p, s1, r, _) in P[s][a]) for a in
                             range(env.action_space.n)]
            # Choose the action with the maximum expected value
            best_action = np.argmax(action_values)
            # Update the policy
            if old_action != best_action:
                policy_stable = False
            new_subdict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, best_action: 1.0}
            policy[s] = new_subdict

        ittr += 1

    return V, policy


def first_visit_mc_prediction(env, policy, num_episodes, gamma):
    """
    Performs the first-visit Monte Carlo prediction algorithm to estimate the state value function.

    Parameters:
        env (gym.Env): The environment.
        policy (dict): The policy to follow. It maps states to action probabilities.
        num_episodes (int): The number of episodes to run.
        gamma (float): The discount factor.

    Returns:
        V (np.array): The estimated state value function.

    """
    # Initialize the state value function and state counters
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)

    # Convert policy dictionary to a numpy array
    policy_array = np.zeros((env.observation_space.n, env.action_space.n))
    for state, action_probs in policy.items():
        for action, prob in action_probs.items():
            policy_array[state][action] = prob

    # Loop for each episode
    for i_episode in range(num_episodes):
        # Generate an episode using the given policy
        episode = []
        state = env.reset()
        done = False
        while not done:
            # Choose an action based on the policy and current state
            if isinstance(state, tuple):
                state_value = state[0]
            else:
                state_value = state
            action = np.random.choice(np.arange(env.action_space.n), p=policy_array[state_value])
            # Take the chosen action and observe the next state, reward, and done flag
            next_state_tuple = env.step(action)
            next_state, reward, done, _, info = next_state_tuple
            # Append the (state, action, reward) tuple to the episode
            episode.append((state_value, action, reward))
            state = next_state

        # Loop for each step of the episode
        states_in_episode = set(tuple(x[0]) if isinstance(x[0], tuple) else x[0] for x in episode)
        for state in states_in_episode:
            # Find the index of the first visit to the state in the episode
            first_visit_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # Calculate the return for the state by summing the discounted rewards
            G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode[first_visit_idx:])])
            # Update the state counter and state value estimate
            N[state] += 1
            V[state] += (G - V[state]) / N[state]

        # Print the state values after each episode
        print(f"Iteration {i_episode + 1}:")
        print([round(value, 5) for value in V.tolist()])

    return V


def every_visit_mc_prediction(env, policy, num_episodes, gamma):
    """
    Performs the every-visit Monte Carlo prediction algorithm to estimate the state value function.

    Parameters:
        env (gym.Env): The environment.
        policy (dict): The policy to follow. It maps states to action probabilities.
        num_episodes (int): The number of episodes to run.
        gamma (float): The discount factor.

    Returns:
        V (np.array): The estimated state value function.
    """

    # Initialize the state value function and state counters
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)

    # Convert policy dictionary to a numpy array
    policy_array = np.zeros((env.observation_space.n, env.action_space.n))
    for state, action_probs in policy.items():
        for action, prob in action_probs.items():
            policy_array[state][action] = prob

    # Loop for each episode
    for i_episode in range(num_episodes):
        # Generate an episode using the given policy
        episode = []
        state = env.reset()
        done = False
        while not done:
            # Choose an action based on the policy and current state
            if isinstance(state, tuple):
                state_value = state[0]
            else:
                state_value = state
            action = np.random.choice(np.arange(env.action_space.n), p=policy_array[state_value])
            # Take the chosen action and observe the next state, reward, and done flag
            next_state, reward, done, _, _ = env.step(action)
            # Append the (state, action, reward) tuple to the episode
            episode.append((state_value, action, reward))
            state = next_state

        # Loop for each step of the episode
        states_in_episode = set(tuple(x[0]) if isinstance(x[0], tuple) else x[0] for x in episode)
        for state in states_in_episode:
            # Collect all returns for the current state in the episode
            returns = [x[2] * (gamma ** i) for i, x in enumerate(episode) if x[0] == state]
            # Calculate the total return for the state
            G = sum(returns)
            # Update the state counter and state value estimate
            N[state] += len(returns)
            V[state] += (G - V[state]) / N[state]

        # Print the state values after each episode
        print(f"Iteration {i_episode + 1}:")
        print([round(value, 5) for value in V.tolist()])

    return V


def format_policy(policy):
    direction_map = {
        0: "LEFT",
        1: "DOWN",
        2: "RIGHT",
        3: "UP"
    }
    result = "Optimal Policy: "
    states = list(policy.keys())
    for i, state in enumerate(states):
        optimal_actions = [direction_map[action] for action, value in policy[state].items() if value == 1.0]
        if optimal_actions:
            result += f"{state}: {', '.join(optimal_actions)}"
            if i != len(states) - 1:
                result += ", "
    return result.rstrip()


############################### custom maps : you have to use them according to the problem
custom_map_1 = ["HFSFFFFG"]

custom_map_2 = ["SFFFF",
                "HHHFF",
                "FFFFH",
                "FFFFF",
                "FFFFG"]

custom_map_3 = ["SFFFF",
                "HFFFF",
                "HFFFF",
                "HFFFF",
                "GFFFF"]

custom_map_4 = ["FFFSFFF",
                "FHHHHFF",
                "FFFFFFF",
                "HFFFFFF",
                "FGFFFFF"]

custom_map_5 = ["HFSFFFFG"]

custom_map_6 = ["HFSFFFFG",
                "HFFFFFFF",
                "HFFFFFFF"]

custom_map_7 = ["SFFFF",
                "FFFFH",
                "HHFFF",
                "HFFFH",
                "FFFFG"]

custom_map_8 = ["HFFSFFH",
                "FFFFFFF",
                "FFFFFFF",
                "GFFHFFG"]
#############################
if __name__ == "__main__":
    # # part 1
    # map_1 = custom_map_1
    # env_1 = gym.make("FrozenLake-v1", render_mode="human", desc=map_1, is_slippery=False)
    # env_1 = ModifyRewards(
    #     env_1, custom_map=map_1, hole_reward=-0.1, goal_reward=1, move_reward=-0.1
    # )
    # env_1.reset()
    # env_1.render()
    # discount_factor_1, discount_factor_2, discount_factor_3, discount_factor_4 = 1, 0.9, 0.5, 0.1
    # V_1, policy_1 = policy_iteration(env_1, map_1, theta=0.0001, discount_factor=0.5)
    # do_policy(env_1, policy_1, episdoes=5)
    # plot_policy_arrows(policy_1, map_1)

    # # part 2
    # map_2 = custom_map_2
    # env_2 = gym.make("FrozenLake-v1", render_mode="human", desc=map_2, is_slippery=False)
    # env_2 = ModifyRewards(
    #     env_2, custom_map=map_2, hole_reward=-4, goal_reward=10, move_reward=-0.9
    # )
    # env_2.reset()
    # env_2.render()
    # discount_factor_1, discount_factor_2, discount_factor_3, discount_factor_4 = 1, 0.9, 0.5, 0.1
    # V_2, policy_2 = policy_iteration(env_2, map_2, theta=0.0001, discount_factor=0.9)
    # do_policy(env_2, policy_2, episdoes=5)
    # plot_policy_arrows(policy_2, map_2)

    # part 3
    # map_3 = custom_map_3
    # env_3 = gym.make("FrozenLake-v1", render_mode="ansi", desc=map_3, is_slippery=False)
    # env_3 = ModifyRewards(
    #     env_3, custom_map=map_3, hole_reward=-5, goal_reward=5, move_reward=-0.5
    # )
    # env_3.reset()
    # env_3.render()
    # V_3, policy_3 = policy_iteration(env_3, map_3, theta=0.0001, discount_factor=0.9)
    # do_policy(env_3, policy_3, episdoes=5)
    # plot_policy_arrows(policy_3, map_3)

    # part 4
    # map_4 = custom_map_4
    # env_4 = gym.make("FrozenLake-v1", render_mode="ansi", desc=map_4, is_slippery=False)
    # env_4 = ModifyRewards(
    #     env_4, custom_map=map_4, hole_reward=-5, goal_reward=5, move_reward=-0.5
    # )
    # env_4.reset()
    # env_4.render()
    # V_4, policy_4 = policy_iteration(env_4, map_4, theta=0.0001, discount_factor=0.9)
    # do_policy(env_4, policy_4, episdoes=5)
    # plot_policy_arrows(policy_4, map_4)

    # part 5
    # map_5 = custom_map_5
    # env_5 = gym.make("FrozenLake-v1", render_mode="ansi", desc=map_5, is_slippery=False)
    # move_reward_1, move_reward_2, move_reward_3, move_reward_4 = -4, -2, 0, 2
    # env_5 = ModifyRewards(
    #     env_5, custom_map=map_5, hole_reward=-3, goal_reward=7, move_reward=move_reward_1
    # )
    # env_5.reset()
    # env_5.render()
    # V_5, policy_5 = policy_iteration(env_5, map_5, theta=0.0001, discount_factor=0.9)
    # do_policy(env_5, policy_5, episdoes=5)

    # part 6
    # map_6 = custom_map_6
    # env_6 = gym.make("FrozenLake-v1", render_mode="human", desc=map_6, is_slippery=False)
    # move_reward_1, move_reward_2, move_reward_3, move_reward_4 = -4, -2, 0, 2
    # env_6 = ModifyRewards(
    #     env_6, custom_map=map_6, hole_reward=-3, goal_reward=7, move_reward=move_reward_4
    # )
    # env_6.reset()
    # env_6.render()
    # V_6, policy_6 = policy_iteration(env_6, map_6, theta=0.0001, discount_factor=0.9)
    # do_policy(env_6, policy_6, episdoes=5)

    # part 7
    # map_7 = custom_map_7
    # env_7 = gym.make("FrozenLake-v1", render_mode="ansi", desc=map_7, is_slippery=True)
    # env_7 = ModifyRewards(
    #     env_7, custom_map=map_7, hole_reward=-2, goal_reward=50, move_reward=-1
    # )
    # env_7.reset()
    # env_7.render()
    # V_7, policy_7 = policy_iteration(env_7, map_7, theta=0.0001, discount_factor=0.9)
    # do_policy(env_7, policy_7, episdoes=5)
    # 
    # num_episode_1, num_episode_2 = 500, 5000
    # gamma = 0.9
    # V_MC = first_visit_mc_prediction(env_7, policy_7, num_episode_1, gamma)
    # V_MCC = every_visit_mc_prediction(env_7, policy_7, num_episode_1, gamma)
    # plot_state_value(V_MC, map_7)
    # plot_policy_arrows(policy_7, map_7)
    # plot_policy_terminal(policy_7, map_7)

    # part 8
    # map_8 = custom_map_8
    # env_8 = gym.make("FrozenLake-v1", render_mode="ansi", desc=map_8, is_slippery=True)
    # env_8 = ModifyRewards(
    #     env_8, custom_map=map_8, hole_reward=-2, goal_reward=50, move_reward=-1
    # )
    # env_8.reset()
    # env_8.render()
    # policy_8_left = get_policy_direction("left", map_8)
    # policy_8_right = get_policy_direction("right", map_8)
    # policy_8_down = get_policy_direction("down", map_8)
    #
    # do_policy(env_8, policy_8_right, episdoes=5)
    #
    # num_episode = 1000
    # gamma = 0.9
    # V_MC = first_visit_mc_prediction(env_8, policy_8_right, num_episode, gamma)
    # #V_MCC = every_visit_mc_prediction(env_8, policy_8_left, num_episode, gamma)
    # print(V_MC)
    # plot_state_value(V_MC, map_8)
    # plot_policy_arrows(policy_8_right, map_8)
    # plot_policy_terminal(policy_8_right, map_8)

    time.sleep(2)
