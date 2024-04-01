from flask import Flask, request
from dev_twillio_app import ConversationManager  # Assuming your class is in this module

app = Flask(__name__)

@app.route("/answer", methods=['POST'])
def answer():
    manager = ConversationManager()
    twiml_response = manager.answer_call()
    return twiml_response

if __name__ == "__main__":
    app.run(debug=True)