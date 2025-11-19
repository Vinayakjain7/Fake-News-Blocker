from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        text = request.form["news_text"]
        data = vectorizer.transform([text])
        result = model.predict(data)[0]
        prediction = "Fake" if result == 1 else "Real"

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)

