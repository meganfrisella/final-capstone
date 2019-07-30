import pickle
import numpy as np


def grab_voice_data():
    """
    grabs voice data from voice_data.p

    parameters:
    ----------
    None

    returns:
    --------
    [np.array] of shape (10,4,44100*4)
        for 10 person, 4 recordings each, with 4 seconds @ 44100 Hz sampling rate
    """
    f = open("voice_data.p", "rb")
    out = pickle.load(f)
    f.close
    return out


def get_data(voice_database):
    """returns tensor (backprop=false) of shape (3,44100*4)
    where idx-0 is the good spec
    idx-1 is the other good spec
    and idx-2 is the bad spec
    
    parameters:
    ----------
    [voice_database] np.array of shape (10,4,44100*4)
        the database given by grab_voice_data()
    
    returns:
    --------
    tensor (backprop=false) of shape (3,44100*4)
    """
    gbindex = np.random.choice(list(range(10)), (2))
    while gbindex[0] == gbindex[1]:
        gbindex = np.random.choice(list(range(10)), (2))

    gdata = np.random.choice(list(range(4)), (2))
    bdata = np.random.choice(list(range(4)))
    while gdata[0] == gdata[1]:
        gdata = np.random.choice(list(range(4)), (2))

    # return (gbindex,gdata,bdata)
    return tensor(
        [
            voice_database[gbindex[0], gdata[0]],
            voice_database[gbindex[0], gdata[1]],
            voice_database[gbindex[1], bdata],
        ],
        requires_grad=False,
    )
