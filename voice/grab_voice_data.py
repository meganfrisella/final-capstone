import pickle

def grab_voice_data():
    '''
    grabs voice data from voice_data.p

    parameters:
    ----------
    None

    returns:
    --------
    [np.array] of shape (10,4,44100*4)
        for 10 person, 4 recordings each, with 4 seconds @ 44100 Hz sampling rate
    '''
    f = open('voice_data.p','rb')
    out = pickle.load(f)
    f.close
    return out
