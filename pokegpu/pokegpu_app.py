from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

print("Loading model...")
with open('model/nature_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/nature_desc.pkl', 'rb') as f:
    nature_desc = pickle.load(f)

print("Model loaded! Server starting...")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    entry = data.get('entry', '')

    if not entry:
        return jsonify({'error': 'No entry provided'}), 400

    nature = model.predict([entry])[0]
    proba = model.predict_proba([entry])[0]
    confidence = round(float(max(proba)) * 100, 1)
    desc = nature_desc.get(nature, '')

    return jsonify({
        'nature': nature,
        'confidence': confidence,
        'description': desc
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\nPokéGPU ML Server running at http://localhost:5000")
    print("Open http://localhost:5000 in your browser!")
    print("Keep this window open while using the Pokédex!\n")
    app.run(debug=False, port=5000)
