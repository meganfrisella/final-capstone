import numpy as np
import pickle
from generate_fridge import layer_image
from generate_fridge import propose_regions
from generate_fridge import parse_food

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

class Fridge:
    """
    Fridge object for simulating a FRIDGECOP® fridge
    """
    
    def __init__(self):
        """Initalizes an empty fridge"""
        self.items = []
        self.user = None
        self.fridge = mpimg.imread('fridge.jpg')

        right = [shift for shift in range(30, 400, 80)]
        shelf_coord = [180, 300, 420, 540, 690]  # coordinates of the first, second ... shelves
        self.shift_ls = []  # possible positions for an item
        for shelf in shelf_coord:
            for pos in right:
                self.shift_ls.append([(shelf, pos)])

        self.images, self.roi_images = parse_food()
        
    def open_fridge(self):
        """
        'Opens' the fridge and sets the person authenticator to the Person who opened it
        """
        photo = input("I'm going to take a photo of your face. Is that ok? [y/n]")
        if photo == 'y':
            photo_consent = True
        else:
            self.user = None
            pass
        if photo_consent:
            self.user = TAKE_PHOTO_AND_RETURN_PERSON_OBJECT()
                
    def add_item(self,item_name):
        """
        Adds item(s) to the fridge
        
        Parameters:
        -----------
        item [Union(tuple[Item] or Item)]
            A tuple of Item objects or a single Item object

        Returns:
        --------
        None
        """
        
        if self.user is not None:
            if isinstance(item,tuple) or isinstance(item,list):
                for i in item:
                    i.owner = self.user
                
            if isinstance(item,tuple):
                self.items += [i for i in item]
                for i in item:
                    image = self.images[self.item_names.index(i.name)]
                    self.fridge = layer_image(self.fridge, propose_regions(image), image, self.shift_ls.pop(np.random.randint(len(self.shift_ls))))
                
            if isinstance(item,Item):
                item.owner = self.user
                self.items.append(item)
                
            if isinstance(item,list):
                self.items += item
        if self.user is None:
            pass
        
            
    def take_item(self,item):
        """
        Takes/removes item(s) from the fridge
        
        Parameters:
        -----------
        item [Union(tuple[Item] or Item)]
            A tuple of Item objects or a single Item object
            
        Returns:
        --------
        None or list of thievery (if item belongs to opener or not, respectively)
        """
        
        self.thief = []
        
        if isinstance(item,tuple) or isinstance(item,list):
            for i in item:
                if i.owner != self.user:
                    self.thief.append(f"{self.user} took {i.owner.name}'s {i.name}'")
                else:
                    throwaway = [self.items.remove(i) for i in item][0]
        if isinstance(item,Item):
            if item.owner != self.user:
                self.thief.append(f"{self.user} took {item.owner.name}'s {item.name}'")
            self.items.remove(item)
            
        if self.thief == []:
            return None
        else:
            return self.thief