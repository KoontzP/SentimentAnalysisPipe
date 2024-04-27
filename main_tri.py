from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline
import numpy as np
from scipy.special import softmax


app = Flask(__name__)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


results = None  # Initialize results variable

conversation = []  # List to hold conversation messages


def label_analysis(prompt):
    # Analyze sentiment of the prompt
    sentiment_output = classifier(prompt)[0]
    label = sentiment_output['label']
    return label


def print_sentiment_distribution(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)[::-1]
    output_str = ""
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        output_str += f"{i+1}) {l} {np.round(float(s), 4)}\n"
    return output_str


@app.route("/", methods=["GET", "POST"])
def chat():
    global results
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            conversation.append(prompt)
            sentiment_label = label_analysis(prompt)
            results = sentiment_label  # Assigning the label to results
            conversation.append(sentiment_label)
            sentiment_distribution = print_sentiment_distribution(prompt)
            print(sentiment_distribution)  # Print sentiment distribution
        return render_template("index_tri.html", conversation=conversation, prompt=prompt, results=results)
    return render_template("index_tri.html", conversation=[], prompt="", results=None)


if __name__ == "__main__":
    app.run(debug=True)
