from dlib_models import download_model, download_predictor, load_dlib_models
from dlib_models import models
import numpy as np
import pickle
from camera import take_picture

#download_model()
#download_predictor()
load_dlib_models()

face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]


def run():
    """
    Takes a photo on the computer's camera and detects faces.
    If a face is recognized, it's name is displayed. If not,
    the user is prompted to enter a name and the person is
    added to the database.

    Parameters
    ----------

    Returns
    -------

    """
    image = take_picture()
    descriptors = image_to_descriptors(image)
    return recognize_image(descriptors[0])


def image_to_descriptors(image):
    """
    Detects faces in an image array and generates a
    128-D descriptor vector for each one.

    Parameters
    ----------
    image : np.array[int]
        RGB image-array or 8-bit grayscale

    Returns
    -------
    List[np.array[float]]
        List of descriptor vectors for each detected
        face in the image

    """
    # Detect faces in the image
    detections = list(face_detect(image))

    # Create a descriptor for each face
    descriptors = []

    for face in detections:
        shape = shape_predictor(image, face)
        descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
        descriptors.append(descriptor)

    return descriptors


def image_to_detections(image):
    """
    Detects faces in an image and generates rectangular
    coordinates that describe each one.

    Parameters
    ----------
    image : np.array[int]
        RGB image-array or 8-bit grayscale

    Returns
    -------
    List[rectangle]
        List of coordinates for each detected face.

    """
    return list(face_detect(image))


def recognize_image(desc):
    """


    Parameters
    ----------


    Returns
    -------


    """

    f = open("people.p", "rb")
    people = pickle.load(f)
    f.close()

    diffs = []
    for person in people.values():
        mean_desc = person.mean_facial_descriptor
        difference = np.sqrt(np.sum(np.square(mean_desc-desc)))
        diffs.append((difference, profile))

    diffs = sorted(diffs)
    return diffs[-1][1]
