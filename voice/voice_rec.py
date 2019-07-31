import functions
import train
import numpy as np


def run_phrase_method():
    ans = input("When ready, hit 'Enter' and say 'Python Like You Mean It'.")
    sample = functions.recording_to_sample(duration=3)
    desc = functions.sample_to_desc(sample)
    name = functions.find_phrase_match(desc)
    return name


def run_encoder_method():
    sample = functions.recording_to_sample(duration=5)
    sample = sample[:176400]
    embed = functions.get_embedding(sample).reshape(100,)
    embed /= np.linalg.norm(embed)
    name = functions.find_encoder_match(embed)
    return name


def run_debug_method(sample):
    embed = functions.get_embedding(sample).reshape(100,)
    embed /= np.linalg.norm(embed)
    name = functions.find_encoder_match(embed)
    return name



