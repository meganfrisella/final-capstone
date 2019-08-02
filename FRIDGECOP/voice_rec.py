import numpy as np
from microphone import record_audio
import pickle

f = open("model2.p", "rb")
model = pickle.load(f)
f.close()

f = open("mean_and_std.p", "rb")
mean, std = pickle.load(f)
f.close()


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
    return model(dft).data


def find_match(emb):
    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    matches = []
    for person in people.values():
        mean_emb = person.mean_vocal_descriptor
        similarity = np.dot(emb, mean_emb)
        matches.append((profile, similarity))

    matches = sorted(matches, key=lambda item: item[1])
    return matches[-1][0]
