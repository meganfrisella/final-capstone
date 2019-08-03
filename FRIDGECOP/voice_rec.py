import numpy as np
from microphone import record_audio
import pickle
import autoencoder

with np.load("model_parameters.npz") as file:
    weight1 = file["weight1"]
    bias1 = file["bias1"]
    weight2 = file["weight2"]
    bias2 = file["bias2"]
    weight3 = file["weight3"]
    bias3 = file["bias3"]

with open("mean_and_std.p", mode="rb") as opened_file:
    mean, std = pickle.load(opened_file)


def run():
    sample = recording_to_sample(duration=5)
    sample = sample[:176400]
    embed = get_embedding(sample).reshape(100,)
    embed /= np.linalg.norm(embed)
    name = find_match(embed)
    return name


def recording_to_sample(duration=3):
    """
    Records audio and converts it to a list of samples with
    a sampling rate of 44100Hz

    Parameters
    ----------
    duration, int
        duration of audio recording

    Returns
    -------
    np.ndarray
        list of samples (voltages) from audio recording

    """
    frames, sample_rate = record_audio(duration)
    audio_data = np.hstack([np.frombuffer(i, np.int16) for i in frames])

    return audio_data


def get_embedding(sample):
    dft = np.abs(np.fft.rfft(sample))
    dft -= mean
    dft /= std
    model = autoencoder.Autoencoder(88201, 100)
    model.load_parameters(weight1, bias1, weight2, bias2, weight3, bias3)
    return model(dft).data


def find_match(emb):
    with open("people.p", mode="rb") as opened_file:
        people = pickle.load(opened_file)

    matches = []
    for person in people:
        mean_emb = person.mean_vocal_descriptor
        similarity = np.dot(emb, mean_emb)
        matches.append((person, similarity))

    matches = sorted(matches, key=lambda item: item[1])
    return matches[-1][0]
