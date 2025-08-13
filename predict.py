import joblib
import sys
import re
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "depression_model.pkl")
vectorizer_path = os.path.join(base_dir, "models", "vectorizer.pkl")

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# If no argument, go interactive
if len(sys.argv) > 1:
    text_input = " ".join(sys.argv[1:])
else:
    text_input = input("Enter text: ")

text_input = clean_text(text_input)

X_input = vectorizer.transform([text_input])
prob = model.predict_proba(X_input)[0][1]
prediction = 1 if prob >= 0.3 else 0  # Lower threshold

probability = model.predict_proba(X_input)[0][1]

print(f"Prediction: {'Depressed' if prediction == 1 else 'Not Depressed'}")
print(f"Confidence: {probability:.4f}")
