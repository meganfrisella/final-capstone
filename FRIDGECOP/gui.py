from tkinter import *
from import_tests import Model
from import_tests import ClassifyingModel
import objects as ob
import tkinter as tk
from PIL import Image, ImageTk

class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):
        
        self.fridge = ob.Fridge()
        
        self.open = False
        
        # changing the title of our master widget      
        self.master.title("FRIDGECOP")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating the buttons
        openFridgeButton = Button(self, text="Open Fridge", command = lambda: openFridgeButton.openFridgeFunc(self.fridge,addItemButton,takeItemButton,closeFridgeButton,seeFridgeButton,printFridgeButton))
        addItemButton = Button(self, text="Add Item to Fridge",command = lambda: addItemButton.addItemFunc(self.fridge))
        takeItemButton = Button(self, text="Take item from Fridge",command = lambda: takeItemButton.takeItemFunc(self.fridge))
        closeFridgeButton = Button(self, text="Close Fridge", command = lambda: closeFridgeButton.closeFridgeFunc(self.fridge,addItemButton,takeItemButton,openFridgeButton,seeFridgeButton,printFridgeButton))
        seeFridgeButton = Button(self, text="See Fridge",command = lambda: seeFridgeButton.seeFridgeFunc(self.fridge))
        printFridgeButton = Button(self, text="Print Fridge",command = lambda: printFridgeButton.printFridgeFunc(self.fridge))
        resetButton = Button(self, text="Reset Fridge",command = lambda: resetButton.resetFunc(openFridgeButton,addItemButton,takeItemButton,closeFridgeButton,seeFridgeButton,printFridgeButton))
        
        #disappearButton = Button(self, text = "Disappear", command = lambda: disappearButton.DAB())

        # placing the button on my window
        resetButton.place(x=400, y=10)
        openFridgeButton.place(x=50,y=50)

        image = Image.open("fridgecop.png")
        photo = ImageTk.PhotoImage(image,master=root)
        label = tk.Label(root, image=photo)
        label.image = photo
        label.place(x = 300,y=25)
                
        
        #addItemButton.place(x=50,y=100)
        #takeItemButton.place(x=50,y=150)
        #closeFridgeButton.place(x=50,y=200)
        #seeFridgeButton.place(x=50,y=250)
        #printFridgeButton.place(x=50,y=300)
        

def openFridgeFunc(self,fridge,add,take,close,see,print_f):
    if app.open:
        print("Fridge is already open")
        pass
    else:
        fridge.open_fridge()
        self.place_forget()
        app.open = True
        add.place(x=50,y=100)
        take.place(x=50,y=150)
        close.place(x=50,y=50)
        see.place(x=50,y=200)
        print_f.place(x=50,y=250)
    
Button.openFridgeFunc = openFridgeFunc

def resetFunc(self,op,add,take,close,see,print_f):
    """
    (self,op,add,take,close,see,print_f)
    """
    print("Resetting Fridge")
    app.fridge = ob.Fridge()
    add.place_forget()
    take.place_forget()
    see.place_forget()
    print_f.place_forget()
    close.place_forget()
    op.place(x=50,y=50)
    app.open = False
    
Button.resetFunc = resetFunc

def closeFridgeFunc(self,fridge,add,take,op,see,print_f):
    if not app.open:
        print("!! Fridge is already closed")
    else:
        self.place_forget()
        fridge.close_fridge()
        
        add.place_forget()
        take.place_forget()
        see.place_forget()
        print_f.place_forget()
        op.place(x=50,y=50)
        app.open = False

Button.closeFridgeFunc = closeFridgeFunc

def addItemFunc(self,fridge):
    item = input("What would you like to put in the fridge? ")
    fridge.add_item(item)
    
Button.addItemFunc = addItemFunc

def takeItemFunc(self,fridge):
    item = input("What would you like to put in the fridge? ")
    fridge.take_item(item)

Button.takeItemFunc = takeItemFunc

def printFridgeFunc(self,fridge):
    ob.print_fridge(fridge)
    
Button.printFridgeFunc = printFridgeFunc

def seeFridgeFunc(self,fridge):
    fridge.show_fridge()
    
Button.seeFridgeFunc = seeFridgeFunc


root = Tk()


#size of the window
root.geometry("500x300")

app = Window(root)
root.mainloop() 