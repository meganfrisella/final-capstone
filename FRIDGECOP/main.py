import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from update_fridge import remove_item, layer_image, propose_regions, parse_food, generate_fridge
from objects import Fridge

def parse_labels():
    with open('food_labels_raw.txt', mode="r") as var:
        categories = var.read().splitlines()
    cat_set = set(categories)
    with open('indiv_labels.txt', mode="r") as f:
        labels = f.read().splitlines()
    cat_dict = {}
    for category in cat_set:
        cat_dict[category] = []
    for i in range(len(categories)):
        cat_dict[categories[i]].append(labels[i])
    return categories, labels, cat_dict

def main():
    categories, labels, cat_dict = parse_labels()
    k = input("Welcome to FRIDGECOP. What is the password?")
    if k.lower() != "plymi":
        print("That is the wrong password. Goodbye")
        return
    fridge = Fridge()
    while True:
        user = input("Hello! Welcome to FRIDGECOP. What is your name?")
        fridge.user = user
        j = input("Would you like to: \n1. Generate random items \n2. Add an item to the fridge\n"
              "3. Remove an item from the fridge\n4. Just display the fridge\n")
        if j == "1":
            num = input("How many items would you like to generate?")
            if type(int(num)) != int:
                print("I'm sorry. That is not a number")
                continue
            fridge.random_fridge(int(num))
            fridge.show_fridge()
            continue
        elif j == "2":
            print(set(categories))
            print()
            cat = input("Which of these would you like to add?")
            if cat not in cat_dict:
                print("Sorry, that is not a category")
                continue
            print(cat_dict[cat])
            print()
            item = input("Which of these items would you like to add?")
            if item not in fridge.item_names:
                print("Sorry, that is not a valid item")
                continue
            fridge.add_item(item, manual = True)
            fridge.show_fridge()
            continue
        elif j == '3':
            print([i.name for i in fridge.scanned_items])
            item = input("What item would you like to remove? Keep in mind that FRIDGECOP is watching...")
            item_obj = None
            for fr_item in fridge.scanned_items:
                if fr_item.name == item:
                    item_obj = fr_item
            if item_obj is None:
                print("I'm sorry. That is not an item in the fridge.")
                continue
            if item_obj.owner != fridge.user:
                confirm = input("That is not your item. Are you sure you want to take this out? (Y/N)")
                if confirm.lower() == "y":
                    fridge.take_item(item, manual = True)
                    continue
                elif confirm.lower() == "n":
                    print("What a kind user.")
                    continue
                else:
                    print("That is not a valid answer.")
                    continue
            else: fridge.take_item(item, manual = True)
            fridge.show_fridge()
            continue
        elif j == "4":
            user_items = []
            for items in fridge.scanned_items:
                if items.owner == fridge.user:
                    user_items.append(items.name)
            print("Hello " + fridge.user)
            print("These are your current items in the fridge.")
            if len(user_items) != 0: print(user_items)
            else: print("You have no items.")
            print("These are your stolen items:")
            if len(fridge.thievery[fridge.user]) != 0: print(fridge.thievery[fridge.user])
            else: print("\tYou have no stolen items.")
            fridge.thievery[fridge.user] = []
            continue
        else:
            print("I'm sorry. That is not a valid command")






if __name__ == "__main__":
    main()
