import numpy as np
import matplotlib.pyplot as plt
from update_fridge import remove_item, layer_image, propose_regions, parse_food
import face_rec


class Person: 
    '''Person identity object for FRIDGECOP
    Parameters:
    name [string]
        The "Firstname Lastname" of the person
    vocal_desc [np.array of shape(1,100)]
        The mean vocal descriptor of the person
    facial_desc [np.array of shape(1,128)]
        The mean facial descriptor of the person
    '''
    
    def __init__(self, name, vocal_desc, facial_desc):
        '''Initializes the Person'''
        self.name = name            
        self.mean_vocal_descriptor = vocal_desc        
        self.mean_facial_descriptor = facial_desc

    def __repr__(self):
        return "FCP: {}".format(self.name)

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

        self.images, self.roi_images, self.item_names = parse_food()

    def show_fridge(self):
        plt.imshow(self.fridge)
        plt.show()

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
            self.user = face_rec.run()
                
    def add_item(self,item_name):
        """
        Adds item(s) to the fridge
        
        Parameters:
        -----------
        item_name [string]
            The string item name

        Returns:
        --------
        None
        """


        if self.user is not None:
            if isinstance(item,tuple) or isinstance(item,list):
                for i in item:
                    i.owner = self.user
                    self.items += [i for i in item]
                    image = self.images[self.item_names.index(i.name)]
                    self.fridge = layer_image(self.fridge, propose_regions(image), image,
                                              self.shift_ls.pop(np.random.randint(len(self.shift_ls))))
                
            if isinstance(item,Item):
                item.owner = self.user
                self.items.append(item)
                image = self.images[self.item_names.index(item.name)]
                self.fridge = layer_image(self.fridge, propose_regions(image), image,
                                          self.shift_ls.pop(np.random.randint(len(self.shift_ls))))

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
                image = self.images[self.item_names.index(i.name)]
                self.fridge = remove_item(image, i.left, i.top)
            throwaway = [self.items.remove(i) for i in item][0]
        if isinstance(item,Item):
            if item.owner != self.user:
                self.thief.append(f"{self.user} took {item.owner.name}'s {item.name}'")
            self.items.remove(item)
            image = self.images[self.item_names.index(item.name)]
            self.fridge = remove_item(image, item.left, item.top)
            
        if self.thief == []:
            return None
        else:
            return self.thief