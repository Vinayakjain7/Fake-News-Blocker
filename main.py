import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = 1   # Fake
df_true["label"] = 0   # Real

df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

print("Dataset loaded. Records:", len(df))
print(df["label"].value_counts())     # SCREENSHOT 1

# ---------------------------------------
# Text Cleaning
# ---------------------------------------
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]','',text)
    return text

df["text"] = df["text"].astype(str).apply(clean)

# ---------------------------------------
# Bar Chart Screenshot
# ---------------------------------------
df["label"].value_counts().plot(kind="bar")
plt.title("Fake (1) vs Real (0) Count")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()                            # SCREENSHOT 2

# ---------------------------------------
# TF-IDF Vectorization
# ---------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# ---------------------------------------
# Train-Test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------------------------------------
# Logistic Regression Model
# ---------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------------------
# Evaluation
# ---------------------------------------
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))   # SCREENSHOT 3
print("\nClassification Report:\n")
print(classification_report(y_test, pred))         # SCREENSHOT 4

# ---------------------------------------
# Confusion Matrix
# ---------------------------------------
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()                                         # SCREENSHOT 5

# ---------------------------------------
# Save Model + Vectorizer
# ---------------------------------------
pickle.dump(model, open("fake_news_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("\nModel and Vectorizer Saved Successfully!")  # SCREENSHOT 6







