import numpy as np
from scipy.stats import beta, dirichlet, entropy
import sounddevice as sd
import time

sample_rate = 44100
duration = 0.5
belief_state = np.full(25, 0.5)
alpha = np.ones(25)
beta_params = np.ones(25)
actions = list(range(25))
well_tempered_frequencies = [220 * 2 ** (n / 12) for n in range(13)]
interval_dict = {
    'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4,
    'P4': 5, 'A4': 6, 'P5': 7, 'm6': 8, 'M6': 9,
    'm7': 10, 'M7': 11, 'P8': 12
}

def generate_tone(frequency):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    envelope = np.exp(-3 * t) * np.sin(2 * np.pi * frequency * t)
    return envelope

def play_interval(semitone_steps):
    base_frequency = np.random.choice(well_tempered_frequencies)
    interval_frequency = base_frequency * (2 ** (semitone_steps / 12))
    base_tone = generate_tone(base_frequency)
    sd.play(base_tone, samplerate=sample_rate)
    sd.wait()
    time.sleep(0.2)
    interval_tone = generate_tone(interval_frequency)
    sd.play(interval_tone, samplerate=sample_rate)
    sd.wait()

def get_user_response(action):
    replays = 0
    semitone_steps = action - 12
    while True:
        play_interval(semitone_steps)
        start_time = time.time()
        user_input = input("Identify the interval (or type 'replay' to hear it again): ")
        elapsed_time = time.time() - start_time
        if user_input.lower() == 'replay':
            replays += 1
        elif user_input in interval_dict:
            correct = (interval_dict[user_input] == abs(semitone_steps))
            return correct, elapsed_time, replays
        else:
            print("Invalid input. Please enter a valid interval name (e.g., 'M3', 'P5').")

def calculate_reward(correct, time_taken, replays):
    return 10 - time_taken - replays if correct else -5

def update_belief(interval_index, correct):
    if correct:
        alpha[interval_index] += 1
    else:
        beta_params[interval_index] += 1
    belief_state[interval_index] = beta.mean(alpha[interval_index], beta_params[interval_index])

def select_action():
    action_probabilities = dirichlet.rvs(belief_state, size=1).flatten()
    action = np.argmax(action_probabilities)
    return action, action_probabilities[action]

def select_action_random():
    action = np.random.choice(actions)
    return action, 1 / len(actions)  # Probability is uniform for random selection

def observe(action, correct, time_taken, replays):
    interval_index = action % 12
    update_belief(interval_index, correct)
    reward = calculate_reward(correct, time_taken, replays)
    return reward

def evaluate_uncertainty():
    return entropy(belief_state)

def train_student(num_questions=10, random_selection=False):
    total_reward = 0
    total_time = 0
    total_replays = 0
    for _ in range(num_questions):
        if random_selection:
            action, prob = select_action_random()
        else:
            action, prob = select_action()
        correct, time_taken, replays = get_user_response(action)
        reward = observe(action, correct, time_taken, replays)
        total_reward += reward
        total_time += time_taken
        total_replays += replays
        print(
            f"Action: {action}, Correct: {correct}, Time Taken: {time_taken:.2f}s, Replays: {replays}, Reward: {reward:.2f}")
    avg_uncertainty = evaluate_uncertainty()
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average time per question: {total_time / num_questions:.2f}s")
    print(f"Average replays per question: {total_replays / num_questions:.2f}")
    print(f"Average uncertainty: {avg_uncertainty:.2f}")

def main():
    print("Running directed multi-armed bandit model:")
    train_student(num_questions=10, random_selection=False)
    print("\nRunning random action selection baseline:")
    train_student(num_questions=10, random_selection=True)

if __name__ == '__main__':
    main()