import functions
import train


def run_phrase_method():
    ready = input("Please repeat the phrase 'Python Like You Mean It'. Hit 'Enter' when ready.")
    sample = functions.recording_to_sample(duration=3)
    freqs = functions.sample_to_freqs(sample)
    name = functions.find_match(freqs)
    return name


def run_encoder_method(path):
    sample = functions.mp3_to_samples(path, duration=4)
    # sample = functions.recording_to_sample(duration=4)
    embed = train.model(sample).data
    return embed


