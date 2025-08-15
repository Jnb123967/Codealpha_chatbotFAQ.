import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Chat loop
if __name__ == "__main__":
    print("FAQ Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print("Bot:", chatbot(user_input))
