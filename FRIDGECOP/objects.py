import numpy as np
import pickle

class Person: 
    '''Person identity object for FRIDGECOP
    Parameters:
    name [string]
        The "Firstname Lastname" of the person
    vocal_desc [Union(tuple,np.array of shape(1,100))]
        The vocal descriptor (or descriptors) of the person
    facial_desc [Union(tuple,np.array of shape(1,128))]
        The facial descriptor (or descriptors) of the person
    '''
    
    def __init__(self, name, vocal_desc, facial_desc):
        '''Initializes the Person'''
        self.name = name
        
        if isinstance(vocal_desc,tuple):
            self.vocal_descriptors = [i for i in vocal_desc]
            
        if isinstance(vocal_desc,np.ndarray) or isinstance(vocal_desc,list):
            self.vocal_descriptors = vocal_desc
            
        self.mean_vocal_descriptor = np.mean(self.vocal_descriptors,axis=0)
        
        if isinstance(facial_desc,tuple):
            self.facial_descriptors = [i for i in facial_desc]
            
        if isinstance(facial_desc,np.ndarray) or isinstance(facial_desc,list):
            self.facial_descriptors = facial_desc
            
        self.mean_facial_descriptor = np.mean(self.facial_descriptors,axis=0)

    def __repr__(self):
        return "FCP: {} ∆ V:{}, F:{}".format(self.name,len(self.vocal_descriptors),len(self.facial_descriptors))
    
    def add_vocal_descriptor(self,vocal_desc,database):
        """Adds a vocal descriptor to the person's list
        
        Parameters:
        -----------
        vocal_desc [np.array] of shape (100,)
            The vocal to be added
            
        database [dict]
            The database dict of 'name':Person
        """
        self.vocal_descriptors.append(vocal_desc)
        self.mean_vocal_descriptor = np.mean(self.vocal_descriptors,axis = 0)
        with open('database.p','wb') as f:
            pickle.dump(database,f)
    
    def add_facial_descriptor(self,facial_desc,database):
        """
        Adds a facial descriptor to the person's list
        
        Parameters:
        -----------
        facial_desc [np.array] of shape (128,)
            The facial descriptor to be added
        
        database [dict]
            The database dict of 'name':Person
        """
        self.facial_descriptors.append(facial_desc)
        self.mean_facial_descriptor = np.mean(self.facial_descriptors,axis = 0)
        with open('database.p','wb') as f:
            pickle.dump(database,f)
            
    def vocal_match(self,vocal_desc_to_be_matched,cutoff):
        """
        Returns boolean if the to_be_matched vocal descriptor matches this Person
        
        Parameters:
        -----------
        vocal_desc_to_be_matched [np.array of shape (100,)]
            the vocal descriptor to be matched
            
        cutoff [float]
            The cutoff value of the < operation
        
        Returns:
        --------
        [boolean]
            
        """
        return np.sum((self.mean_vocal_descriptor - vocal_desc_to_be_matched)**2) ** (1 / 2) < cutoff
    
    def facial_match(self,facial_desc_to_be_matched,cutoff):
        """
        Returns boolean if the to_be_matched facial descriptor matches this Person
        
        Parameters:
        -----------
        facial_desc_to_be_matched [np.array of shape (100,)]
            the facial descriptor to be matched
            
        cutoff [float]
            The cutoff value of the < operation
        
        Returns:
        --------
        [boolean]
            
        """
        return np.sum((self.mean_vocal_descriptor - facial_desc_to_be_matched)**2) ** (1 / 2) < cutoff