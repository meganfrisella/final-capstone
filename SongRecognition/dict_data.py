import pickle

def import_database():
    """Imports databse of song footprints
    
    parameters:
    -----------
    none
    
    returns:
    --------
    dictionary of database
    
     """
    f = open("database.p", "rb")
    database = pickle.load(f)
    f.close()
    return database


def import_test():
    """import length-9 tuple of 9-second test samples

    Parameters:
    -----------
    None

    Returns:
    --------
    length-9 tuple of 9-second test samples"""
    f = open("test.p", "rb")
    test_tuple = pickle.load(f)
    f.close()
    return test_tuple

def import_dictionaries(string):
    """imports dictionaries
    
    Parameters:
    -----------
    string [type: string]
        The dictionary that you want:
    
    int_to_title : integer to title values (ex: {0: "Another One Bites The Dust by Queen"})
    
    int_to_pathstring : integer to pathstring values (ex: {0: "another"})
    
    Returns:
    --------
    [dict]
        [dictionary for your input]"""
    
    f = open("save.p", "rb")
    int_to_title = pickle.load(f)
    int_to_pathstring = pickle.load(f)
    f.close()
    
    if string =="int_to_title":
        return int_to_title
    if string =="int_to_pathstring":
        return int_to_pathstring
    
    else:
        print("WRONG INPUT, CHECK DOCSTRING")
    