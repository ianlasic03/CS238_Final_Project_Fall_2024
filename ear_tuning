import numpy as np
import time
from scipy.stats import entropy
import sounddevice as sd
from transitions import run_simulation

class POMDP:
    def __init__(self):
        """
        Initialize the POMDP for interval assessment.
        """
        # states is a dictionary of 13 intervals with key as interval and value is skill level for that interval
        self.states = ['bad', 'ok', 'good']

        # action for each interval
        self.actions = list(range(25))

        # generate transition probs using beta distributions
        self.transition_probs = run_simulation(3, 25, 10000, 20)
        print(f"transition_probs: {self.transition_probs}")

        # initialize observation probabilities
        self.observation_probs = np.zeros((2, len(self.states), len(self.actions)))
        for i, state in enumerate(self.states):
            if state == 'good':
                self.observation_probs[1, i, :] = 0.9  # High chance of correct observation
                self.observation_probs[0, i, :] = 0.1  # Low chance of incorrect observation
            elif state == 'ok':
                self.observation_probs[1, i, :] = 0.6
                self.observation_probs[0, i, :] = 0.4
            elif state == 'bad':
                self.observation_probs[1, i, :] = 0.3
                self.observation_probs[0, i, :] = 0.7
        # Normalize observation_probs along the observation axis
        self.observation_probs /= self.observation_probs.sum(axis=0, keepdims=True)
        print(f"Observation_probs: {self.observation_probs}")
        # initialize belief state
        self.belief = np.full(len(self.states), 1 / len(self.states))

        # Initialize reward structure
        self.rewards = {
            True: 10,  # Correct answer
            False: -10  # Incorrect answer
        }

        self.sample_rate = 44100
        self.duration = 0.5
        self.well_tempered_frequencies = [220 * 2 ** (n / 12) for n in range(13)]
        self.interval_dict = {
            'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4,
            'P4': 5, 'A4': 6, 'P5': 7, 'm6': 8, 'M6': 9,
            'm7': 10, 'M7': 11, 'P8': 12
        }

    """
        Bayes rule to update belief using discrete state filter
    """

    def update(self, belief, P, action, observation):
        states = P.states
        transition = P.transition_probs
        observations = P.observation_probs


        # New belief
        b_prime = np.zeros_like(belief)

        for i_prime, s_prime in enumerate(states):
            prob_o = observations[observation, i_prime, action]
            b_prime[i_prime] = prob_o * sum(
                (transition[i, action, i_prime] * belief[i] for i, s in enumerate(states))
            )

        # Normalize the updated belief
        if np.sum(b_prime) == 0:
            b_prime = np.ones_like(belief)

        normalize_constant = np.sum(b_prime)
        return b_prime / normalize_constant

    def information_gain(self, belief, P, action):
        states = P.states
        observations = P.observation_probs

        expected_KL_divergence = 0.0
        possible_observations = [0, 1]
        for observation in possible_observations:
            updated_belief = self.update(belief, P, action, observation)
            KL_divergence = entropy(updated_belief, belief)
            # observed probaility of taking obsservation o and action a from each state
            observation_prob = sum(observations[observation, i, action] * belief[i] for i in range(len(states)))
            expected_KL_divergence += observation_prob * KL_divergence
        return expected_KL_divergence

    def select_best_action(self, belief, P, actions):
        information_gains = [self.information_gain(belief, P, action) for action in actions]
        value = np.argmax(information_gains)
        if value > 12:
            value = value - 12
        else:
            value = value
        reversed_dict = {value: key for key, value in self.interval_dict.items()}
        print(f"Action: {reversed_dict.get(value)}")
        return np.argmax(information_gains)

    def random_action(self, P):
        action = np.random.choice(P.actions)
        if action > 12:
            action = action - 12
        reversed_dict = {value: key for key, value in self.interval_dict.items()}
        print(f"Action: {reversed_dict.get(action)}")
        return action

    def evaluate_uncertainty(self, belief):
        return entropy(belief, np.ones(len(belief)) / len(belief))

    def run_experiment(self, strategy, num_flashcards, P, belief):
        initial_uncertainty = self.evaluate_uncertainty(belief)
        print(f"inital belief: {belief}")
        for _ in range(num_flashcards):
            if strategy == "info_gain":
                action = self.select_best_action(belief, P, P.actions)
            elif strategy == "random":
                action = self.random_action(P)
            observation = self.get_user_response(action)
            belief = self.update(belief, P, action, observation)
            print(f"updated belief: {belief}")
        final_uncertainty = self.evaluate_uncertainty(belief)
        return initial_uncertainty, final_uncertainty

    def generate_tone(self, frequency):
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        envelope = np.exp(-3 * t) * np.sin(2 * np.pi * frequency * t)
        return envelope

    def play_interval(self, semitone_steps):
        base_frequency = np.random.choice(self.well_tempered_frequencies)
        interval_frequency = base_frequency * (2 ** (semitone_steps / 12))
        base_tone = self.generate_tone(base_frequency)
        sd.play(base_tone, samplerate=self.sample_rate)
        sd.wait()
        time.sleep(0.2)
        interval_tone = self.generate_tone(interval_frequency)
        sd.play(interval_tone, samplerate=self.sample_rate)
        sd.wait()

    def get_user_response(self, action):
        semitone_steps = action - 12
        while True:
            self.play_interval(semitone_steps)
            user_input = input("Identify the interval (or type 'replay' to hear it again): ")
            if user_input in self.interval_dict:
                correct = (self.interval_dict[user_input] == abs(semitone_steps))
                return int(correct)
            else:
                print("Invalid input. Please enter a valid interval name (e.g., 'M3', 'P5').")

# Example usage
def main():
    P = POMDP()
    num_flashcards = 20

    # Run experiment with information gain strategy
    info_gain_initial, info_gain_final = P.run_experiment(
        "info_gain", num_flashcards, P, P.belief.copy()
    )
    print(f"Initial uncertainty (info gain): {info_gain_initial}")
    print(f"Final uncertainty (info gain): {info_gain_final}")

    # Run experiment with random action strategy
    random_initial, random_final = P.run_experiment(
        "random", num_flashcards, P, P.belief.copy()
    )
    print(f"Initial uncertainty (random): {random_initial}")
    print(f"Final uncertainty (random): {random_final}")


if __name__ == "__main__":
    main()
