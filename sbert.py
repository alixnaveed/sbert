from flask import Flask, render_template

from sentence_transformers import SentenceTransformer

application = Flask(__name__)

# Specify the path to the model folder
model_folder = "./sentence-transformers_bert-base-nli-mean-tokens"

# Load the pre-trained SBERT model from the specified folder
model = SentenceTransformer(model_folder)

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
