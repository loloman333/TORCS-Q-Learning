from __future__ import annotations
import pickle

import numpy as np
import math
import matplotlib.pyplot as plt

from types import SimpleNamespace

def moving_avg(array, windows_size):
    windows_size = round(windows_size)
    result = [np.nan for _ in range(0, round(windows_size/2))]
    moving_sum = sum(array[:windows_size])
    result.append(moving_sum / windows_size)
    for i in range(windows_size, len(array)):
        moving_sum += (array[i] - array[i - windows_size])
        result.append(moving_sum / windows_size)
    return result

class QLearner:

    EXPORT_PATH: str = "./ex.port"
    QTABLE_EXPORT_PATH: str = "./qtable"

    def __init__(self, num_states: int, num_actions: int, epsilon: float, alpha: float, gamma: float, epsilon_change: float, epsilon_min) -> None:

        self.num_states = num_states    
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_change = epsilon_change
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma

        self.keep_learning = True
        self.last_state = None
        self.last_action = None
        self.score = 0
        
        self.stats = SimpleNamespace()
        self.stats.scores = []
        self.stats.best_score = 0
        self.stats.epsilon_values = []
        self.stats.stop_learning = []
        self.stats.actions = []
        self.stats.observations = []
        self.q_table = np.zeros((num_states, num_actions))
        
    def __str__(self) -> str:
        return f"""
            Q-Learner

            Number of States: {self.num_states}
            Number of Actions: {self.num_actions}
            Q-Table Entries: {self.num_actions * self.num_states}

            Alpha: {self.alpha}
            Gamma: {self.gamma}
            (Current) Epsilon: {self.epsilon}
            Epsilon Change per Episode: {self.epsilon_change}
            Min Epsilon: {self.epsilon_min}

            Number of Episodes in stats: {len(self.stats.scores)}
            Total Average Score: {np.average(self.stats.scores)}
            Still Learning: {self.keep_learning}
        """

    def __repr__(self) -> str:
        return self.__str__()
    
    def _export(self) -> None:
        with open(self.EXPORT_PATH, "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def _import() -> QLearner:
        with open(QLearner.EXPORT_PATH, "rb") as file:
            return pickle.load(file)

    def export_qtable(self) -> None:
        with open(self.QTABLE_EXPORT_PATH, "wb") as file:
            pickle.dump(self.q_table, file)

    def import_qtable(self) -> None:
        with open(self.QTABLE_EXPORT_PATH, "rb") as file:
            self.q_table = pickle.load(file)

    def q_table_string(self):
        string = f"  + {'     '.join([f'{i:3}' for i in range(0, self.num_actions)])}\n"

        for i in range(0, self.num_states):
            string += f"{i:3}  "
            for j in range(0, self.num_actions):
                string += f"{self.q_table[i][j]:6.3f}  "
            string += "\n"

        return string.rstrip("\n")

    def stop_learning(self):
        self.keep_learning = False
        self.stats.stop_learning.append(len(self.stats.scores))

    def get_substate(self, value, state_count, state_length):
        tempered = False
        if state_count % 2 != 0:
            state_count *= 2 #TODO half value instead?
            state_length /= 2
            tempered = True

        substate = value / state_length
        substate = math.trunc(substate) - 1 if substate < 0 else math.trunc(substate)
        substate = math.trunc(state_count / 2) - 1 if substate >= math.trunc(state_count / 2) else substate
        substate = -math.trunc(state_count / 2) - 1 if substate <= -math.trunc(state_count / 2) else substate
        substate += math.trunc(state_count / 2)
        substate = math.trunc(substate / 2) if tempered else substate

        return substate

    def combine_substates(self, substates, substate_counts):
        state = 0
        mulitplier = 1
        for index, substate in enumerate(substates):
            state += substate * mulitplier
            mulitplier *= substate_counts[index]

        return state

    def get_state(self, observation):
        raise NotImplementedError

    def epsilon_deacy(self):
        #"""
        if self.epsilon > self.epsilon_min: 
            self.epsilon += self.epsilon_change
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        """
        if (len(self.stats.scores) > 100):

            if self.score >= np.average(self.stats.scores[-100:-1]):
                self.epsilon -= self.epsilon_change
            else:
                self.epsilon += self.epsilon_change
        """

    def end_episode(self):

        if self.score > self.stats.best_score:
            self.stats.best_score = self.score

        self.stats.epsilon_values.append(self.epsilon)
        self.stats.scores.append(self.score)

        #if len(self.stats.scores) % 10 == 0 and len(self.stats.scores) != 0:
        print(f"Episode: {len(self.stats.scores)}, Total Best: {self.stats.best_score}, Last 10 Best: {max(self.stats.scores[-10:])}, Last 10 Averge: {np.average(self.stats.scores[-10:]):.1f}, Last 10 Min: {min(self.stats.scores[-10:]):.1f}, Epsilon: {self.epsilon:.3f}")
        #    self._export()

        self.epsilon_deacy()

        self.last_state = None
        self.last_action = None
        self.score = 0
            
    def plot_stats(self):

        fig, ax = plt.subplots()
        ax.set_xlabel('Episodes')
        ax2 = ax.twinx()

        # Score
        ax.plot(self.stats.scores, linestyle=' ', marker='.', color='#887aff', label="Scores")
        ax.set_ylabel('Score', color="blue")
        ax.tick_params(axis='y', colors="blue")

        # Moving Average Score
        ax.plot(moving_avg(self.stats.scores, len(self.stats.scores) / 10), color="blue", label="Moving Average")
        
        # Epsilon
        ax2.plot(self.stats.epsilon_values, color="orange")
        ax2.set_ylabel('Epsilon', color="orange")
        ax2.tick_params(axis='y', colors="orange")

        # Stop Learning
        #for x in self.stats.stop_learning:
        #    plt.axvline(x, color = 'red', label = 'Stopped learning')
        #    plt.text(x, 5, "Stopped\nlearning", rotation=0, verticalalignment='center')

        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.show()

    def plot_observations_actions(self):

        observations_0s_x = []
        observations_0s_y = []
        observations_1s_x = []
        observations_1s_y = []

        for index, action in enumerate(self.stats.actions):
            if action == 0:
                observations_0s_x.append(self.stats.observations[index][0])
                observations_0s_y.append(self.stats.observations[index][1])
            else:
                observations_1s_x.append(self.stats.observations[index][0])
                observations_1s_y.append(self.stats.observations[index][1])                             

        plt.plot(observations_0s_x, observations_0s_y, color="orange", label="0s", linestyle=" ", marker=".")
        plt.plot(observations_1s_x, observations_1s_y, color="purple", label="1s", linestyle=" ", marker=".")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.show()

        plt.plot(observations_0s_x, observations_0s_y, color="orange", label="0s", linestyle=" ", marker=",")
        plt.plot(observations_1s_x, observations_1s_y, color="purple", label="1s", linestyle=" ", marker=",")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.show()

    def policy(self, observation) -> int:

        state = self.get_state(observation)

        random = np.random.uniform()

        if self.keep_learning and random < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[state])
            
        self.last_state = state
        self.last_action = action

        #self.stats.observations.append(observation)
        #self.stats.actions.append(action)

        return action

    def learn(self, observation, reward, score):
        self.score += score
        if not self.keep_learning: return

        current_q = self.q_table[self.last_state][self.last_action]
        max_future_q = np.max(self.q_table[self.get_state(observation)])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[self.last_state][self.last_action] = new_q

