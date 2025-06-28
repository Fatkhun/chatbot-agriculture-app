from transformers import BertTokenizerFast, TFBertForSequenceClassification
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import random
import json

app = Flask(__name__)

#Load the model and tokenizer
model_path = "BINUS/indoAgricultureBert"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

# Label untuk klasifikasi (sesuaikan dengan model Anda)
# --- Load and Prepare Data ---
def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        file = json.load(f)
    return file   

filename = "model/dataset.json"
intents = load_json_file(filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_suggestions')
def get_suggestions():
    # Define tags to exclude from suggestions
    EXCLUDED_TAGS = ['noanswer', 'greeting', 'perkenalan', 'penutup']
    
    # Default suggestions in case of error
    DEFAULT_SUGGESTIONS = [
        "Bagaimana cara menyemai benih cabai rawit?",
        "Ciri-ciri benih cabai berkualitas dan kondisi lahan siap tanam?",
        "Apa sejarah tanaman cabai di Indonesia dan manfaatnya bagi kesehatan?",
        "Bagaimana cara melakukan penanaman cabai rawit?"
    ]
    
    try:
        suggestions = []
        
        for intent in intents['intents']:
            tag = intent.get('tag', '')
            
            # Skip if tag is in excluded list or if no patterns exist
            if tag.lower() in EXCLUDED_TAGS or not intent.get('patterns'):
                continue
                
            # Add up to 2 examples from each intent
            examples = intent['patterns'][:2]
            suggestions.extend(examples)
        
        # Shuffle and limit to 4 suggestions
        random.shuffle(suggestions)
        suggestions = suggestions[:4]
        
        # If we ended up with no suggestions, use defaults
        if not suggestions:
            return jsonify({'suggestions': DEFAULT_SUGGESTIONS})
            
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        print(f"Error getting suggestions: {str(e)}")
        return jsonify({'suggestions': DEFAULT_SUGGESTIONS})

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        data = request.get_json()
        user_message = data['message']
        
        # Tokenisasi input
        inputs = tokenizer(user_message, return_tensors="tf", truncation=True, padding=True,max_length=512)
        
       # Prediksi kelas
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])
        
        print(f"Confidence: {confidence}")
        
        if confidence < 0.7:
            return jsonify({'response': "Maaf, saya tidak bisa menjawab pertanyaan itu. Bisakah Anda mencoba mengulanginya dengan cara lain?"})
        else:
            response = random.choice(intents['intents'][predicted_class_idx]['responses'])
            print(f"CHATBOT: {response}")
            return jsonify({'response': response})
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({'response': "Sorry, I encountered an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)