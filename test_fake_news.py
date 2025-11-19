import pickle

# Load saved model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sample text for testing
sample_text = """Breaking: Prime Minister announces new economic reforms to boost GDP."""

# Transform & predict
x = vectorizer.transform([sample_text])
prediction = model.predict(x)[0]

print("Input Text:", sample_text)
print("Prediction:", "FAKE" if prediction == 1 else "REAL")




