from transformers import pipeline
from flask import Flask, request, jsonify, render_template

classifier = pipeline("text-classification", model="KoontzP/Finetuned-sentiment-model")


def label_analysis(prompt):
    pred = classifier([prompt])
    print(pred[0])
    label = pred[0]['label']
    number = int(label.split('_')[1])
    if number == 0:
        return "sadness"
    elif number == 1:
        return "joy"
    elif number == 2:
        return "love"
    elif number == 3:
        return "anger"
    elif number == 4:
        return "fear"
    else:
        return "surprise"


app = Flask(__name__)

conversation = []  # List to hold conversation messages


@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            conversation.append(prompt)
            labels_output = label_analysis(prompt)
            print(labels_output)
            conversation.append(labels_output)
        # Render the template with the conversation and the latest prompt
        return render_template("index.html", conversation=conversation, prompt=prompt)

    # Render the template initially with an empty conversation and no prompt
    return render_template("index.html", conversation=[], prompt="")


# Define a route to render the form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)