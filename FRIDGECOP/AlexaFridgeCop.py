from flask import Flask
from flask_ask import Ask, statement, question
from import_tests import Model, ClassifyingModel
from objects import Fridge, check_fridge
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
    return question("Fridge Cop on duty. What's the password?")

@ask.intent("WhatsInMyFridgeIntent")
def open_fridge():
    person = voice_rec.run()
    items = check_fridge(fridge, person)
    return statement(items)

@ask.intent("StopIntent")
def cancel_intent():
    return statement("Alright, adios!")

if __name__ == "__main__":
    # with open('model7.p', 'rb') as f:
    #     USF_model = pickle.load(f)
    app.run(debug=True)
