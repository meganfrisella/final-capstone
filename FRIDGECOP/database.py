import voice_rec
import face_rec
import pickle
import numpy as np
from camera import take_picture
import objects


def initialize_database():
    people = []
    with open("people.p", mode="wb") as opened_file:
        pickle.dump(people, opened_file)


def add_person(num_photos=3, num_recordings=5):
    name = input("What is your name (First Last)? ")

    print("We will take 5 five-second recordings of your voice. Please speak consistently for the "
          "entire time.")

    ans = input("Hit 'Enter' when ready")

    samples = np.zeros((num_recordings, 176400))

    for i in range(num_recordings):
        sample = voice_rec.recording_to_sample(duration=5)
        samples[i] = sample[:176400]

    mean_vocal_emb = np.mean(voice_rec.get_embedding(samples), axis=0)
    mean_vocal_emb /= np.linalg.norm(mean_vocal_emb)

    print("We will take 3 pictures of your face. Please move your head slightly in between photos.")

    ans = input("Hit 'Enter' when ready")

    descs = []
    for i in range(num_photos):
        print(f"Taking photo {i+1}")
        image = take_picture()
        desc = face_rec.image_to_descriptors(image)[0]
        descs.append(desc)
    mean_face_desc = np.mean(descs, axis=0)

    with open("people.p", mode="rb") as opened_file:
        people = pickle.load(opened_file)

    people.append(objects.Person(name, mean_vocal_emb, mean_face_desc))

    with open("people.p", mode="wb") as opened_file:
        pickle.dump(people, opened_file)
