import pickle
import numpy as np
from torch import tensor

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
    f.close()
    return out

def get_data(batch_size,test=False):
    '''returns tensor (backprop=false) of shape (batch_size,3,1,1000,85)
    where idx-0 is the good spec
    idx-1 is the other good spec
    and idx-2 is the bad spec
    
    parameters:
    ----------
    batch_size [int]
        batch size
    
    returns:
    --------
    tensor (backprop=false) of shape (3,1,1000,85)
    '''
    
    out = np.zeros((batch_size,3,1,1000,85))
    
    
    if test:
        for idx in range(batch_size):

            gbindex= np.random.choice(list(range(5)),(2))
            while gbindex[0] == gbindex[1]:
                 gbindex = np.random.choice(list(range(5)),(2))

            gdata = np.random.choice(list(range(10)),(2))
            bdata = np.random.choice(list(range(10)))
            while gdata[0] == gdata[1]:
                 gdata = np.random.choice(list(range(10)),(2))

            #return (gbindex,gdata,bdata)
            out[idx] = [test_data[gbindex[0],gdata[0]], test_data[gbindex[0],gdata[1]], test_data[gbindex[1],bdata]]
        return (tensor(out[:,0]).float().to(device),tensor(out[:,1]).float().to(device),tensor(out[:,2]).float().to(device))
    else:
        for idx in range(batch_size):

            gbindex= np.random.choice(list(range(5)),(2))
            while gbindex[0] == gbindex[1]:
                 gbindex = np.random.choice(list(range(5)),(2))

            gdata = np.random.choice(list(range(42)),(2))
            bdata = np.random.choice(list(range(42)))
            while gdata[0] == gdata[1]:
                 gdata = np.random.choice(list(range(42)),(2))

            #return (gbindex,gdata,bdata)
            out[idx] = [train_data[gbindex[0],gdata[0]], train_data[gbindex[0],gdata[1]], train_data[gbindex[1],bdata]]
        return (tensor(out[:,0]).float().to(device),tensor(out[:,1]).float().to(device),tensor(out[:,2]).float().to(device))