import numpy as np
from microphone import record_audio
import librosa
import pickle

f = open("model.p", "rb")
model = pickle.load(f)
f.close()

f = open("mean_and_std.p", "rb")
mean, std = pickle.load(f)
f.close()


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


def mp3_to_sample(path, duration=4):
    """
    Converts an mp3 file to a list of samples with
    a sampling rate of 44100Hz

    Parameters
    ----------
    path, string
        path to an mp3 file

    duration, int
        duration of sample, in seconds

    Returns
    -------
    np.ndarray
        list of samples (voltages) from audio recording

    """
    samples, fs = librosa.load(path, sr=44100, mono=True, duration=duration)
    return samples


def sample_to_desc(sample, cutoff=0.9999, bin_size=12):
    """
    Generates a list of top frequencies for
    a given sample of audio.

    Parameters
    ----------
    sample, np.ndarray
        an audio sample

    cutoff, np.float
        the percent of coefficients to filter out

    Returns
    -------
    np.ndarray
        an array of amplitudes for corresponding k-values

    """
    N = len(sample)
    L = N / 44100
    c = np.abs(np.fft.rfft(sample))

    c = c[:-1]

    min_amp = np.sort(c)[int(cutoff*len(c))]

    new_c = np.copy(c)
    new_c[c < min_amp] = 0

    new_c = new_c.reshape((int(len(new_c)/bin_size), -1))
    new_c = np.mean(new_c, axis=1).reshape(new_c.shape[0])

    new_c = new_c / np.linalg.norm(new_c)
    """
    fig, ax = plt.subplots()
    ax.plot(new_c)
    ax.set_xlim(0, 250)
    """

    # k = np.arange(N//2+1)
    # freq = k / L

    return new_c


def find_phrase_match(desc, cutoff=0.5):
    desc = desc / np.linalg.norm(desc)

    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    matches = []
    for profile in people.values():
        mean_desc = profile.mean_desc
        mean_desc = mean_desc / np.linalg.norm(mean_desc)
        similarity = np.dot(desc, mean_desc)
        matches.append((profile.name, similarity))

    matches = sorted(matches, key=lambda item: item[1])
    return matches[-3:]


def find_encoder_match(emb, cutoff=0.8):

    f = open("encoder.p", "rb")
    people = pickle.load(f)
    f.close()

    matches = []
    for profile in people.values():
        mean_emb = profile.mean_desc
        similarity = np.dot(emb, mean_emb)
        matches.append((profile.name, similarity))

    matches = sorted(matches, key=lambda item: item[1])
    return matches[-1][0]


def get_embedding(sample):
    dft = np.abs(np.fft.rfft(sample))
    dft -= mean
    dft /= std
    return model(dft).data

