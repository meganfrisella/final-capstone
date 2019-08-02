import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from update_fridge import remove_item, layer_image, propose_regions, parse_food
import face_rec
import math

def items_close(item1, item2):
    """
    Checks if two items are relatively close to account for object detection error
    :param item1: Item object
    :param item2: Item object
    :return: boolean
        True or False if the distance is within 30 pixels
    """
    dist = math.sqrt((item1.top - item2.top) ** 2 + (item1.left - item2.left) ** 2)
    return dist <= 30


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

    def vocal_match(self,vocal_desc_to_be_matched, cutoff):
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


class Item:
    def __init__(self, left, top, name, category, owner):
        self.left = left
        self.top = top
        self.name = name
        self.category = category
        self.owner = owner


class Fridge:
    """
    Fridge object for simulating a FRIDGECOP® fridge
    """
    
    def __init__(self):
        """Initalizes an empty fridge"""
        self.items = []
        self.thief = []
        self.user = None
        self.fridge = mpimg.imread('fridge.jpg')

        right = [shift for shift in range(30, 400, 80)]
        shelf_coord = [180, 300, 420, 540, 690]  # coordinates of the first, second ... shelves
        self.shift_ls = []  # possible positions for an item of (top, left)
        for shelf in shelf_coord:
            for pos in right:
                self.shift_ls.append(tuple(shelf, pos))

        self.images, self.roi_images, self.item_names, self.categories = parse_food()

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
                
    def add_item(self, item_name):
        """
        Adds item(s) to the fridge
        
        Parameters:
        -----------
        item_name : String or List[String]
            The item name as one String or a List of Strings

        Returns:
        --------
        None
        """

        if self.user is not None:
            if isinstance(item_name, list):
                for i in item_name:
                    if i in self.item_names:
                        if len(self.shift_ls) == 0:  # Checks if there are no available spaces in the fridge
                            print("FRIDGECOP says the fridge is full")
                            return
                        position = self.shift_ls.pop(np.random.randint(len(self.shift_ls)))
                        image = self.images[self.item_names.index(i)]
                        self.fridge = layer_image(self.fridge, propose_regions(image), image, position)
                    else:
                        print("FRIDGECOP does not recognize this food item")
                
            if isinstance(item_name, str):
                if item_name in self.item_names:
                    if len(self.shift_ls) == 0:  # Checks if there are no available spaces in the fridge
                        print("FRIDGECOP says the fridge is full")
                        return
                    position = self.shift_ls.pop(np.random.randint(len(self.shift_ls)))
                    image = self.images[self.item_names.index(item_name)]
                    self.fridge = layer_image(self.fridge, propose_regions(image), image, position)
                else:
                    print("FRIDGECOP does not recognize this food item")

        if self.user is None:
            pass

    def take_item(self, item_name):
        """
        Takes/removes item(s) from the fridge
        
        Parameters:
        -----------
        item_name : String or List[String]
            The item name as one String or a List of Strings
            
        Returns:
        --------
        None or list of thievery (if item belongs to opener or not, respectively)
        """
        
        if isinstance(item_name, list):
            for name in item_name:
                item_obj = None
                for fr_item in self.items:
                    if fr_item.name == name:
                        item_obj = fr_item
                if item_obj is None:
                    print("FRIDGECOP does not recognize that food item")
                    return
                if item_obj.owner != self.user:
                    self.thief.append(f"{self.user} took {item_obj.owner.name}'s {name}'")
                image = self.images[self.item_names.index(name)]
                self.fridge = remove_item(image, item_obj.left, item_obj.top)
                self.shift_ls.remove((item_obj.top, item_obj.left))
        if isinstance(item_name, Item):
            name = item_name
            item_obj = None
            for fr_item in self.items:
                if fr_item.name == name:
                    item_obj = fr_item
            if item_obj is None:
                print("FRIDGECOP does not recognize that food item")
                return
            if item_obj.owner != self.user:
                self.thief.append(f"{self.user} took {item_obj.owner.name}'s {name}'")
            image = self.images[self.item_names.index(name)]
            self.fridge = remove_item(image, item_obj.left, item_obj.top)
            self.shift_ls.remove((item_obj.top, item_obj.left))
            
        if len(self.thief) == 0:
            return None
        else:
            return self.thief

    def close_fridge(self):
        self.new_scan = SCAN_FRIDGE() #returns list of Item objects
        #self.scanned_items
        
        self.added_items = []
        for i in self.new_scan:
            if i not ISCLOSE(() self.scanned_items)) #!!!!!!!!:
                self.added_items.append(i)
                
        for i in self.added_items:
            i.owner = self.user
        
        self.user = None