def get_triplet_for_megan(data):
    '''
    gets triplets for megan
    
    Parameters:
    -----------
    data [np.array] of some shape??
    
    Returns:
    --------
    [np.array] of shape (3,176400)
        The triplets of 4-second samples
            where idx-0 is the good spec
            idx-1 is the other good spec
            and idx-2 is the bad spec
    '''
    out = np.zeros(3,176400)
    gbindex= np.random.choice(list(range(data.shape[0])),(2))
    while gbindex[0] == gbindex[1]:
         gbindex = np.random.choice(list(range(data.shape[0])),(2))

    gdata = np.random.choice(list(range(data.shape[1])),(2))
    bdata = np.random.choice(list(range(data.shape[1])))
    while gdata[0] == gdata[1]:
         gdata = np.random.choice(list(range(data.shape[1])),(2))

    #return (gbindex,gdata,bdata)
    out[idx] = [data[gbindex[0],gdata[0]], data[gbindex[0],gdata[1]], data[gbindex[1],bdata]]