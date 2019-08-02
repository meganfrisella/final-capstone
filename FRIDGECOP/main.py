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
    if k.lower() != "python like you mean it":
        print("That is the wrong password. Goodbye")
        return
    j = input("Correct! Would you like to: \n 1. Generate a random fridge (Generate) \n 2. Add an item to the fridge (Add item)\n"
          "3. Remove an item from the fridge (Remove item)\n 4. Just check your fridge (Check fridge)")
    if j.lower() == "generate":
        num = input("How many items would you like to generate in your fridge?")
        generate_fridge(int(num))
    elif j.lower() == "add item":
        fridge = Fridge()
        fridge.open_fridge()
        print(set(categories))
        print()
        cat = input("Which of these would you like to add?")
        print(cat_dict[cat])
        print()
        item = input("Which of these items would you like to add?")
        fridge.add_item(item)
        fridge.show_fridge()


if __name__ == "__main__":
    main()
