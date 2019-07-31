import functions
import train


def run_phrase_method():
    ans = input("When ready, hit 'Enter' and say 'Python Like You Mean It'.")
    sample = functions.recording_to_sample(duration=3)
    desc = functions.sample_to_desc(sample)
    name = functions.find_match(desc)
    return name


def run_encoder_method(path):
    sample = functions.mp3_to_samples(path, duration=4)
    # sample = functions.recording_to_sample(duration=4)
    embed = train.model(sample).data
    return embed



