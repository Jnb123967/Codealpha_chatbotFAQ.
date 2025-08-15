import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyngrok import ngrok

# Load FAQ data
with open("faq.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chatbot(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    return answers[idx]

# Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = chatbot(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")
    app.run(port=5000)
