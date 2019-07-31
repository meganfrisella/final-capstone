import numpy as np
from microphone import record_audio
import librosa
import pickle


def recording_to_sample(duration=5):
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


def mp3_to_samples(path, duration=4):
    """
    Converts an mp3 file to a list of samples with
    a sampling rate of 44100Hz

    Parameters
    ----------
    path, string
        path to an mp3 file

    Returns
    -------
    np.ndarray
        list of samples (voltages) from audio recording

    """
    samples, fs = librosa.load(path, sr=44100, mono=True, duration=duration)
    samples *= 2**15
    return samples


def sample_to_freqs(sample, num_freqs=100):
    """
    Generates a list of top frequencies for
    a given sample of audio.

    Parameters
    ----------
    sample, np.ndarray
        an audio sample

    num_freqs, int
        the number of top frequencies to be included

    Returns
    -------
    list, len=num_freqs
        the top frequencies in the sample

    """
    c = np.abs(np.fft.rfft(sample))

    N = len(sample)
    L = N / 44100
    k = np.arange(N // 2 + 1)
    freqs = np.round(k / L, 3)

    coef_freq = list(zip(c, freqs))
    top_n = sorted(coef_freq)[-num_freqs:]

    return [freq for coef, freq in top_n]


def find_match(freq, cutoff=0.5):
    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    freq = np.array(freq)/np.linalg.norm(freq)

    for profile in people.values():
        mean_freq = profile.mean_freq
        mean_freq = np.array(mean_freq)/np.linalg.norm(mean_freq)

        similarity = np.dot(freq, mean_freq)
        print(profile.name, similarity)
