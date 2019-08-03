from flask import Flask
from flask_ask import Ask, statement, question
from objects import Fridge, check_fridge
import voice_rec
import requests
import time
import unidecode
import json
import voice_rec

app = Flask(__name__)
ask = Ask(app, '/')

fridge = None

@app.launch
def start_skill():
    global fridge
    fridge = Fridge()
    return question("Fridge Cop on duty. What's the password?")

@ask.intent("WhatsInMyFridgeIntent")
def open_fridge():
    person = voice_rec.run()
    items = check_fridge(fridge, person)
    return statement(f"Hello, {person}! you have {items} in your fridge.")

@ask.intent("StopIntent")
def cancel_intent():
    return statement("Alright, adios!")

if __name__ == "__main__":
    app.run(debug=True)