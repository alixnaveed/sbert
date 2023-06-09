from flask import Flask, render_template

from sentence_transformers import SentenceTransformer

application = Flask(__name__)

# Load the pre-trained SBERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Example sentences
sentences = [
    "I love coding",
    "Machine learning is fascinating",
    "The cat is sitting on the mat"
]

# Compute sentence embeddings
embeddings = model.encode(sentences)

@application.route('/')
def home():
    data = zip(sentences, embeddings)
    return render_template('index.html', data=data)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080)
