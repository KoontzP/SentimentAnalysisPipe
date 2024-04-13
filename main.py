from transformers import pipeline
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, render_template

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def label_analysis(prompt):
    batch = tokenizer(prompt, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch)
        label_ids = torch.argmax(outputs.logits, dim=1)
        labels = [model.config.id2label[label_ids] for label_ids in label_ids.tolist()]
    return labels




app = Flask(__name__)

conversation = []  # List to hold conversation messages

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            conversation.append(prompt)  # Add the user's prompt to the conversation
            # Generate the chatbot's response based on the user's prompt
            labels_output = label_analysis(prompt)
            conversation.append(f"Labeled as {labels_output[0]}.")  # Add the chatbot's response to the conversation
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