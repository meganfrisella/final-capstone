from flask import Flask
from flask_ask import Ask, statement, question
from objects import Fridge, check_fridge
from use_scanned_fridge import Model, ClassifyingModel
import requests
import time
import unidecode
import json
import voice_rec

app = Flask(__name__)
ask = Ask(app, '/')

fridge = None

@app.route('/')
def homepage():
    return "Currently running AlexaFridgeCop.py"

@ask.launch
def start_skill():
    global fridge
    fridge = Fridge()
    fridge.open_fridge()
    fridge.add_item(["crab", "gum", "chicken plate", "quiche", "mushroom", "kimchi", "caviar"])
    return question("Fridge Cop on duty. What's the password?")

@ask.intent("WhatsInMyFridgeIntent")
def open_fridge():
    person = voice_rec.run()
    print(person)
    items = check_fridge(fridge, person)
    print(items)
    return statement(f"Hello, {person}! you have {items} in your fridge.")

@ask.intent("StopIntent")
def cancel_intent():
    return statement("Alright, adios!")

if __name__ == "__main__":
    app.run(debug=True)