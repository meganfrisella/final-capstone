import numpy as np
import matplotlib.mlab as mlab
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from pathlib import Path
from microphone import record_audio

def mp3_to_sample(path):
    """converts an mp3 file to 16-bit @44.1kHz samples
    
    !!requires the librosa library!!
    import librosa
    
    
    Parameters:
    -----------
    path: [string]
    The path of the mp3
    ex: str(Path("Audio/Believe-The_Bravery.mp3"))
    
    Returns:
    --------
    [np.array]
        [16-bit samples at 44.1kHz]"""
    samples, fs = librosa.load(path, sr=44100, mono=True)
    return samples*(2**15)

def mic_to_sample(length):
    """records audio and returns the 16-bit 44.1kHz samples
    
    !!requires ryan's microphone library
    from microphone import record_audio
    
    
    Parameters:
    -----------
    length: [float]
    The length of the recording in seconds
    
    Returns:
    --------
    [np.array]
        [16-bit samples at 44.1kHz]"""

    frames, sample_rate = record_audio(length)
    audio_data = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    return audio_data

def sample_to_spectrogram(sample):
    """converts 16-bit 44.1kHz audio samples to a spectrogram
    
    Parameters:
    -----------
    sample: [np.array]
    
    16-bit 44.1kHz audio sample array
    
    Returns:
    --------
    [mlab.specgram object]
    
        [Returns S, freqs, times of mlab.specgram object (see ex)]
        
    Ex: 
    
    S, freqs, times = mlab.specgram(believe_sample, NFFT=4096, Fs=44100,
                                  noverlap=4096 // 2)"""

    return mlab.specgram(sample, NFFT=4096, Fs=44100,noverlap=4096 // 2)
    #S, freqs, times = mlab.specgram(sample, NFFT=4096, Fs=44100,noverlap=4096 // 2)
    #return np.log(np.clip(S,a_min=10**-20,a_max=None))