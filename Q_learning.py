import itertools
import os
import numpy as np
import torch
import random
from bach import LSTMModel, predict_sequence
from midi_tools import one_hot_to_midi


class BachMelodyQLearning:
    def __init__(self, num_notes=38, num_durations=5):
        self.num_notes = num_notes
        self.num_durations = num_durations

        self.state_space = (self.num_notes * self.num_durations) ** 2
        self.action_space = self.num_notes * self.num_durations

        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        self.hidden_size = 128
        self.output_size = num_notes + num_durations  # Output will be note + duration
        self.input_size = num_notes + num_durations  # Input size is also note + duration

        self.LSTM_model = None

    def set_up_LSTM_model(self, model):
        self.LSTM_model = model

    def generate_vectors(self, length):
        return np.eye(length, dtype=int)

    def initialize_table(self):
        q_table = {}
        # Generate individual arrays
        array1 = self.generate_vectors(38)
        array2 = self.generate_vectors(5)
        array3 = self.generate_vectors(38)
        array4 = self.generate_vectors(5)

        # Create all combinations of the arrays
        combinations = itertools.product(array1, array2, array3, array4)

        # Iterate over combinations
        results = []
        for combination in combinations:
            # Combine the arrays to form a single array (4 concatenated vectors)
            i1 = np.argmax(np.array(combination[0]))
            i2 = np.argmax(np.array(combination[1]))
            i3 = np.argmax(np.array(combination[2]))
            i4 = np.argmax(np.array(combination[3]))
            results.append((i1, i2, i3, i4))

        # Convert results to a numpy array for easier manipulation
        for result in results:
            q_table[result] = {}

        action_combinations = itertools.product(array1, array2)
        combos = []
        for product in action_combinations:
            i1 = np.argmax(product[0])
            i2 = np.argmax(product[1])
            combos.append((i1, i2))

        for s in q_table.keys():
            for a in combos:
                q_table[s][a] = 0
        return q_table

    def get_rewards(self, sequence, action):
        pred_pitch_probs, pred_duration_probs = predict_sequence(sequence)

        pitch = pred_pitch_probs[0][action[0]]
        dur = pred_duration_probs[0][action[1]]

        reward = pitch + dur

        return reward

    def select_action(self, q_table, state):
        first_actions = next(iter(q_table.values())).keys()
        available_actions = list(first_actions)
        best_action = None
        max_q_val = float("-inf")
        if random.random() < self.exploration_rate:
            action = random.choice(available_actions)
            return action
        else:
            for action in available_actions:
                q_val = q_table[state][action]
                if q_val > max_q_val:
                    max_q_val = q_val
                    best_action = action
            return best_action

    def Q_learning_update(self, q_table, state, action, reward):
        q_val = q_table[state][action]

        max_next_q_val = max(q_table[state].values())

        q_val += self.learning_rate * (reward + self.discount_factor * max_next_q_val - q_val)
        q_table[state][action] = q_val


    def train(self, q_table, epochs=10000, sequence_len=32):
        for epoch in range(epochs):
            print(epoch)
            # initialize a random start point
            current_state = (20, 0, 22, 0)

            # Loop to generate the sequence
            for _ in range(sequence_len - 1):
                state = current_state
                action = self.select_action(q_table, state)

                sequence = list(state)
                embeddings = np.zeros((2, 43))
                embeddings[0][sequence[0]], embeddings[0][sequence[1] + 38] = 1, 1
                embeddings[1][sequence[2]], embeddings[1][sequence[3] + 38] = 1, 1
                tensor_sequence = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
                reward = self.get_rewards(tensor_sequence, action)

                # Shift the sequence and append next state
                sequence[0] = sequence[2]
                sequence[1] = sequence[3]
                sequence[2] = action[0]
                sequence[3] = action[1]

                self.Q_learning_update(q_table, state, action, reward)
                current_state = tuple(sequence)
        return q_table

    def generate_melody(self, q_table, sequence_length=32):
        melody_sequence = []
        state = (20, 0, 22, 0)

        train_reward = 0

        for _ in range(sequence_length):
            best_action = max(q_table[state], key=q_table[state].get)
            train_reward += max(q_table[state].values())

            embeddings = np.zeros(43)
            embeddings[best_action[0]], embeddings[best_action[1] + 38] = 1, 1

            melody_sequence.append(embeddings)

            state = list(state)
            state[0] = state[2]
            state[1] = state[3]
            state[2] = best_action[0]
            state[3] = best_action[1]
            state = tuple(state)

        print(f"Reward for trained sequence: {train_reward}")
        return melody_sequence

    def generate_random_melody(self, q_table, sequence_length=32):
        melody_sequence = []
        state = (20, 0, 22, 0)

        random_reward = 0

        first_actions = next(iter(q_table.values())).keys()
        available_actions = list(first_actions)

        for _ in range(sequence_length):
            best_action = random.choice(available_actions)
            random_reward += q_table[state][best_action]

            embeddings = np.zeros(43)
            embeddings[best_action[0]], embeddings[best_action[1] + 38] = 1, 1

            melody_sequence.append(embeddings)

            state = list(state)
            state[0] = state[2]
            state[1] = state[3]
            state[2] = best_action[0]
            state[3] = best_action[1]
            state = tuple(state)
        print(f"Reward for random sequence: {random_reward}")
        return melody_sequence


def main():
    # Initialize and train LSTM model
    model = LSTMModel(input_size=43, hidden_size=128, output_size=43)

    # Load pre-trained LSTM model
    if os.path.exists("lstm_model.pth"):
        model.load_state_dict(torch.load("lstm_model.pth"))
        print("Model loaded successfully!")

    # Initialize Q-learning agent
    q_learning_agent = BachMelodyQLearning()
    q_learning_agent.set_up_LSTM_model(model)

    # Initialize Q-leariing table
    q_table = q_learning_agent.initialize_table()

    # Train Q-learning agent
    q_table = q_learning_agent.train(q_table)

    # Generate melody
    generated_melody = q_learning_agent.generate_melody(q_table)
    print(generated_melody)
    random_melody = q_learning_agent.generate_random_melody(q_table)

    one_hot_to_midi(generated_melody, "melody.mid")

    one_hot_to_midi(random_melody, "random_melody.mid")


if __name__ == '__main__':
    main()
