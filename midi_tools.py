import numpy as np
import mido
from mido import MidiFile


# tick_map = {
#     40: 0,  # Triplet 32nd note
#     60: 1,  # 32nd note
#     80: 2,  # Triplet 16th note
#     90: 3,  # Dotted 32nd note
#     120: 4,  # 16th note
#     160: 5,  # Triplet 8th note
#     180: 6,  # Dotted 16th
#     240: 7,  # 8th note
#     320: 8,  # Triplet quarter note
#     360: 9,  # Dotted 8th note
#     480: 10,  # Quarter note
#     720: 11,   # Dotted quarter
#     960: 12,  # Half note
#     1080: 13,
#     1920: 14,  # Whole note
# }


tick_map = {
   120: 0,  # 16th note
   240: 1,  # 8th note
   360: 2,  # Dotted 8th note
   480: 3,  # Quarter note
   # 960: 4,  # Half note
   1080: 4,  # Half note + 16th
}






# Invert tick_map for duration mapping
inverse_tick_map = {v: k for k, v in tick_map.items()}




class Midi:
   def __init__(self):
       # Define the note names
       note_map = {0: 'A', 1: 'A#', 2: 'B', 3: 'C',
                   4: 'C#', 5: 'D', 6: 'D#', 7: 'E',
                   8: 'F', 9: 'F#', 10: 'G', 11: 'G#'}


       # Generate the MIDI map from D3 (50) to D6 (86)
       midi_map = {i: f"{note_map[i % 12]}_{i // 12}" for i in range(41, 78)}
       midi_map[0] = 'pause'  # Add a pause option


       # Store the mappings
       self.midi_map = midi_map
       self.note_map = {v: k for k, v in midi_map.items()}




def preprocess_top_stave(filepath):
   midi = MidiFile(filepath)
   track = midi.tracks[1]  # Use the first track (top stave)
   events = []
   cur_time = 0
   notes_on = []


   for msg in track:
       if msg.is_meta:
           continue
       cur_time += msg.time
       if msg.type == 'note_on' and msg.velocity > 0:
           notes_on.append((cur_time, msg.note - 21))  # Normalize note to 0 (A0)
       elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
           for i, (start_time, note) in enumerate(notes_on):
               if note == msg.note - 21:
                   events.append((start_time, cur_time, note))
                   notes_on.pop(i)
                   break
   events.sort()
   return events




def events_to_lstm_input(events, seq_len=32, num_notes=38, num_durations=5):
   """
   Converts events into a one-hot representation for LSTM training.


   Args:
       events: List of (start, end, note) tuples.
       seq_len: Length of each sequence.
       num_notes: Number of unique notes (88 notes + 1 pause).


   Returns:
       X: Input sequences (N x seq_len x num_notes).
       y: Output labels (N x num_notes).
   """
   sequences = []
   labels = []
   one_hot_pause = np.zeros(num_notes + num_durations)
   one_hot_pause[0] = 1  # Pause is the first entry


   one_hot_notes = [np.zeros(num_notes) for _ in range(88)]
   for i in range(num_notes - 1):
       one_hot_notes[i][i + 1] = 1  # Note indices start at 1


   # Initialize one-hot encodings for durations
   one_hot_durations = [np.zeros(num_durations) for _ in range(num_durations)]
   for i in range(num_durations - 1):
       one_hot_durations[i][i + 1] = 1  # Duration indices start at 1


   for i in range(len(events) - seq_len):
       sequence = []
       for j in range(seq_len):
           start, end, note = events[i + j]
           duration = end - start


           # Ensure duration is within bounds
           if duration <= 0:
               sequence.append(one_hot_pause)
           else:
               duration_idx = tick_map[duration]  # Cap to max duration
               note_duration = np.concatenate([one_hot_notes[note], one_hot_durations[duration_idx]])
               sequence.append(note_duration)


       next_note = events[i + seq_len][2]  # Predict the note after the sequence
       next_duration = events[i + seq_len][1] - events[i + seq_len][0]  # Duration for the next note


       # Find the corresponding index for the next note and duration
       duration_idx = tick_map[next_duration]
       label = np.concatenate([one_hot_notes[next_note], one_hot_durations[duration_idx]])


       sequences.append(sequence)
       labels.append(label)


   return np.array(sequences), np.array(labels)




class MidiProcessor(Midi):
   def __init__(self, seq_len=32):
       super().__init__()
       self.seq_len = seq_len


   def process_for_lstm(self, filepath):
       events = preprocess_top_stave(filepath)
       X, y = events_to_lstm_input(events, seq_len=self.seq_len)
       return X, y




def one_hot_to_midi(one_hot_embeddings, output_file):
   midi = MidiFile()
   track = mido.MidiTrack()
   midi.tracks.append(track)


   # Add meta messages for clef and key signature
   track.append(mido.MetaMessage('key_signature', key='C'))
   track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))  # Default 4/4 time signature


   velocity = 64  # Default velocity for all notes
   tick_inverse = {v: k for k, v in tick_map.items()}  # Inverse map for tick values


   for embedding in one_hot_embeddings:
       #print(embedding)
       # Extract pitch and duration
       pitch_idx = np.argmax(embedding[:38])
       duration_idx = np.argmax(embedding[38:])


       # Determine note/rest and duration in ticks
       if pitch_idx == 0:  # Rest
           #print(duration_idx)
           tick_duration = tick_inverse.get(duration_idx)  # Default to quarter note if not mapped
           track.append(mido.Message('note_off', velocity=0, time=tick_duration))
           continue


       midi_note = pitch_idx + 41  # Convert to MIDI note number
       tick_duration = tick_inverse.get(duration_idx)


       # Add messages for note_on and note_off
       track.append(mido.Message('note_on', note=midi_note, velocity=velocity, time=0))
       track.append(mido.Message('note_off', note=midi_note, velocity=velocity, time=tick_duration))


   # Save the MIDI file
   midi.save(output_file)
   print(f"MIDI file saved as {output_file}")




if __name__ == "__main__":
   # one_hot_list = [np.eye(104)[np.random.choice(104)] for _ in range(32)]
   # one_hot_to_midi(one_hot_list, "output.mid")
   # Example usage
   midi = Midi()
   print(midi.midi_map)