from flask import Flask
from flask_ask import Ask, statement, question
from main import run
import requests
import time
import unidecode
import json
import random

app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Currently running SongRecognition.py"

@ask.launch
def start_skill():
    return question("Do you want to figure out the name and artist of a song?")

@ask.intent("YesIntent")
def recognize_song():
    song = run()
    if song is 'Sorry, we could not find you song :(':
        return statement(song)
    return statement("You are playing " + song + ".")

@ask.intent("NoIntent")
def no_intent():
    return statement("Alright, peace!")

if __name__ == "__main__":
    app.run(debug=True)