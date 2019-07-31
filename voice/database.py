import pickle
import numpy as np
import functions
import matplotlib.pyplot as plt


def initialize_database():
    people = {}
    f = open('people.p', 'wb')
    pickle.dump(people, f)
    f.close()


class Profile:
    def __init__(self, name, descs):
        self.name = name
        self.descs = descs
        self.mean_desc = np.mean(self.descs, axis=0)

    def __repr__(self):
        return "{} ∆ {}".format(self.name, len(self.descs))

    def add_descriptor(self, desc, people):
        self.descs.append(desc)
        self.mean_desc = np.mean(self.descs, axis=0)
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
    name = input("What is your name (First Last)? ")

    print("We will take 4 recordings of your voice. Please repeat the following phrase when prompted: "
          "'Python Like You Mean It'")

    ans = input("Hit 'Enter' when ready")

    descs = []
    for num in range(num_recordings):
        print(f"Starting audio sample {num+1}.")
        sample = functions.recording_to_sample()
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


def remove_profile(name):
    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    del people[name]

    f = open('people.p', 'wb')
    pickle.dump(people, f)
    f.close()
