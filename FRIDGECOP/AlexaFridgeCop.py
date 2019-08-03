from flask import Flask
from flask_ask import Ask, statement, question
from import_tests import Model, ClassifyingModel
from objects import Fridge, check_fridge, print_fridge
import voice_rec
import pickle

app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Currently running AlexaFridgeCop.py"

@ask.launch
def start_skill():
    return question("Fridge Cop on duty. Password?")

@ask.intent("WhatsInMyFridgeIntent")
def open_fridge():
    with open("fridge.p", mode="rb") as opened_file:
        fridge = pickle.load(opened_file)
    person = voice_rec.run()
    print(check_fridge(fridge, person))
    items = check_fridge(fridge, person)
    return statement(items)

@ask.intent("StopIntent")
def cancel_intent():
    return statement("Okay bye.")

if __name__ == "__main__":
    # with open('model7.p', 'rb') as f:
    #     USF_model = pickle.load(f)
    app.run(debug=True)
