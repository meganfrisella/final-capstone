import functions


def run():
    sample = functions.recording_to_sample()
    freqs = functions.sample_to_freqs(sample)
    name = functions.find_match(freqs)
    return name