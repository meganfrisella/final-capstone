import pickle
import numpy as np
import functions
import matplotlib.pyplot as plt


def initialize_database(type):

    if type is 'phrase':
        people = {}
        f = open('people.p', 'wb')
        pickle.dump(people, f)
        f.close()

    elif type is 'encoder':
        encoder = {}
        f = open('encoder.p', 'wb')
        pickle.dump(encoder, f)
        f.close()


class Profile:
    def __init__(self, name, descs):
        self.name = name
        self.descs = descs
        self.mean_desc = np.mean(self.descs, axis=0)

    def __repr__(self):
        return "{} ∆ {}".format(self.name, len(self.descs))


class EncoderProfile:
    def __init__(self, name, mean_emb):
        self.name = name
        self.mean_desc = mean_emb

    def __repr__(self):
        return self.name


def add_phrase_profile(num_recordings=5):
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
    name = input("What is your name (First Last)? ")

    print("We will take 5 recordings of your voice. Please repeat the following phrase when prompted: "
          "'Python Like You Mean It'")

    ans = input("Hit 'Enter' when ready")

    descs = []
    for num in range(num_recordings):
        print(f"Starting audio sample {num+1}.")
        sample = functions.recording_to_sample(duration=3)
        desc = functions.sample_to_desc(sample)
        descs.append(desc)

    for desc in descs:
        fig, ax = plt.subplots()
        ax.plot(desc)
        ax.set_xlim(0, 250)

    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    people[name] = Profile(name, descs)

    f = open('people.p', 'wb')
    pickle.dump(people, f)
    f.close()


def add_encoder_profile(num_recordings=5):
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
    name = input("What is your name (First Last)? ")

    print("We will take 5 five-second recordings of your voice. Please speak consistently for the "
          "entire time.")

    ans = input("Hit 'Enter' when ready")

    samples = np.zeros((num_recordings, 176400))

    for i in range(num_recordings):
        sample = functions.recording_to_sample(duration=5)
        samples[i] = sample[:176400]

    mean_emb = np.mean(functions.get_embedding(samples), axis=0)
    mean_emb /= np.linalg.norm(mean_emb)

    f = open("encoder.p", "rb")
    people = pickle.load(f)
    f.close()

    people[name] = EncoderProfile(name, mean_emb)

    f = open('encoder.p', 'wb')
    pickle.dump(people, f)
    f.close()


def add_encoder_profile_debug(samples, name):
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

    mean_emb = np.mean(functions.get_embedding(samples), axis=0)
    mean_emb /= np.linalg.norm(mean_emb)

    f = open("encoder.p", "rb")
    people = pickle.load(f)
    f.close()

    people[name] = EncoderProfile(name, mean_emb)

    f = open('encoder.p', 'wb')
    pickle.dump(people, f)
    f.close()


def remove_profile(type, name):

    if type is 'phrase':
        f = open("people.p", "rb")
        people = pickle.load(f)
        f.close()
        del people[name]
        f = open('people.p', 'wb')
        pickle.dump(people, f)
        f.close()

    elif type is 'encoder':
        f = open("encoder.p", "rb")
        people = pickle.load(f)
        f.close()
        del people[name]
        f = open('encoder.p', 'wb')
        pickle.dump(people, f)
        f.close()