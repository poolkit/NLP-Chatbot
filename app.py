from flask import Flask, render_template, url_for, request, jsonify
import load_model

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

def chatbotResponse():
    if request.method == 'POST':
        the_question = request.form['question']
        response = load_model.predict(the_question)
    return jsonify({"response": response })

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    response = load_model.predict(msg)
    return response.capitalize()

if __name__ == "__main__":
    app.run(debug=True)

