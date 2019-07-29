import pickle
import numpy as np
import functions


def initialize_database():
    people = {}
    f = open('people.p', 'wb')
    pickle.dump(people, f)
    f.close()


class Profile:
    def __init__(self, name, freqs):
        self.name = name
        self.freqs = freqs
        self.mean_freq = np.mean(self.freqs, axis=0)

    def __repr__(self):
        return "{} ∆ {}".format(self.name, len(self.freqs))

    def add_descriptor(self, freq, people):
        self.freqs.append(freq)
        self.mean_freq = np.mean(self.freqs, axis=0)
        f = open('people.p', 'wb')
        pickle.dump(people, f)
        f.close()


def add_profile(num_recordings=5):
    """
    Creates a profile for a new person and adds it to the database. A specified
    number of 5-second recordings are taken, during which the person should
    be speaking into the microphone.

    Parameters
    ----------
    num_recordings, np.int
        the number of recordings taken

    Returns
    -------

    """
    name = input("What is your name (First Last)?")

    freqs = []
    for num in range(num_recordings):
        print(f"Starting audio sample {num+1}.")
        sample = functions.recording_to_sample()
        freqs.append(functions.sample_to_freqs(sample))

    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    people[name] = Profile(name, freqs)

    f = open('people.p', 'wb')
    pickle.dump(people, f)
    f.close()
