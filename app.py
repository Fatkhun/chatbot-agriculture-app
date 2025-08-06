from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import torch
import random
from typing import List, Dict, Tuple, Optional

app = Flask(__name__)

# Configuration
DATA_FILE = os.path.join('static', 'assets', 'dataset_rev_updated.json')
MODEL_NAME = "BINUS/indobert-agriculture-qa"
CACHE_DIR = os.path.join('static', 'assets', 'model_cache')
SIMILARITY_THRESHOLD = 0.5  # Threshold for context similarity
MAX_CONTEXT_LENGTH = 512  # Maximum context length for QA model
TOP_K_CONTEXTS = 1       # Number of top contexts to consider

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def download_models() -> Tuple[Optional[AutoModelForQuestionAnswering], 
                              Optional[AutoTokenizer], 
                              Optional[SentenceTransformer]]:
    """Download and cache all models"""
    print("Downloading models...")
    try:
        # QA Model
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR
        )
        qa_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR
        )
        
        # Similarity Model (multilingual for better Indonesian support)
        similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', 
                                             cache_folder=CACHE_DIR)
        
        return qa_model, qa_tokenizer, similarity_model
    except Exception as e:
        print(f"Failed to download models: {e}")
        return None, None, None

def initialize_pipelines() -> Tuple[Optional[pipeline], Optional[SentenceTransformer]]:
    """Initialize all pipelines"""
    print("Initializing pipelines...")
    try:
        qa_model, qa_tokenizer, similarity_model = download_models()
        
        if all([qa_model, qa_tokenizer, similarity_model]):
            qa_pipeline = pipeline(
                'question-answering',
                model=qa_model,
                tokenizer=qa_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            return qa_pipeline, similarity_model
        return None, None
    except Exception as e:
        print(f"Failed to initialize pipelines: {e}")
        return None, None

def load_and_preprocess_data() -> Tuple[List[Dict], List[Dict], Dict[str, Dict]]:
    """Load and preprocess data with efficient structures"""
    print("Loading and preprocessing data...")
    global dataset
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # List to store all paragraphs with their questions and contexts
        paragraphs = []
        # List to store all question-answer pairs for fallback suggestions
        qa_pairs = []
        # Dictionary to store exact question matches {question: {'answer': ..., 'context': ..., 'answer_start': ...}}
        exact_qa_matches = {}
        
        for item in dataset['data']:
            for paragraph in item['paragraphs']:
                context = paragraph['context'].strip()
                
                for qa in paragraph['qas']:
                    if not qa['is_impossible'] and qa['answers']:
                        question = qa['question'].strip()
                        answer = qa['answers'][0]['text'].strip()
                        answer_start = qa['answers'][0]['answer_start']
                        
                        exact_qa_matches[question] = {
                            'answer': answer,
                            'context': context,
                            'answer_start': answer_start
                        }
                        
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'context': context
                        })
                
                paragraphs.append({
                    'context': context,
                    'qas': paragraph['qas']
                })
        
        print(f"Loaded {len(paragraphs)} paragraphs and {len(qa_pairs)} Q&A pairs")
        return paragraphs, qa_pairs, exact_qa_matches
    except Exception as e:
        print(f"Failed to load data: {e}")
        return [], [], {}

# Initialize components
qa_pipeline, similarity_model = initialize_pipelines()
PARAGRAPHS, QA_PAIRS, EXACT_QA_MATCHES = load_and_preprocess_data()

# Precompute question embeddings for similarity search
QUESTION_EMBEDDINGS = None
QUESTIONS = []
CONTEXTS = []

if similarity_model:
    print("Computing question embeddings...")
    # Prepare lists of questions and their corresponding contexts
    for paragraph in PARAGRAPHS:
        for qa in paragraph['qas']:
            QUESTIONS.append(qa['question'])
            CONTEXTS.append(paragraph['context'])
    
    if QUESTIONS:
        QUESTION_EMBEDDINGS = similarity_model.encode(QUESTIONS)
        print(f"Computed embeddings for {len(QUESTIONS)} questions")

def find_most_similar_question(user_question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the most similar question from the dataset to the user's question.
    Returns a tuple of (most_similar_question, context) if found, otherwise (None, None)
    """
    if similarity_model is None or QUESTION_EMBEDDINGS is None or len(QUESTIONS) == 0:
        return None, None
    
    # Encode the user's question
    user_embedding = similarity_model.encode([user_question])
    
    # Calculate similarities
    similarities = cosine_similarity(user_embedding, QUESTION_EMBEDDINGS)[0]
    
    # Get the index of the most similar question
    most_similar_idx = np.argmax(similarities)
    max_similarity = similarities[most_similar_idx]
    
    # Only return if similarity is above threshold
    if max_similarity > SIMILARITY_THRESHOLD:
        return QUESTIONS[most_similar_idx], CONTEXTS[most_similar_idx]
    
    return None, None

def find_most_relevant_context(question: str, top_k: int = TOP_K_CONTEXTS) -> List[str]:
    if not similarity_model or not PARAGRAPHS:
        return [p['context'] for p in PARAGRAPHS[:top_k]]
    
    question_embedding = similarity_model.encode([question])
    
    if not hasattr(find_most_relevant_context, 'context_embeddings'):
        find_most_relevant_context.context_embeddings = similarity_model.encode(
            [p['context'] for p in PARAGRAPHS]
        )
    
    similarities = cosine_similarity(
        question_embedding,
        find_most_relevant_context.context_embeddings
    )[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_contexts = [
        PARAGRAPHS[idx]['context']
        for idx in top_indices
        if similarities[idx] > SIMILARITY_THRESHOLD
    ]
    
    # Jika tak ada konteks yang memenuhi threshold, ambil top_k saja
    if not relevant_contexts:
        relevant_contexts = [PARAGRAPHS[i]['context'] for i in top_indices]
    
    return relevant_contexts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not qa_pipeline or not PARAGRAPHS:
        return jsonify({'response': "Maaf, sistem belum siap. Silakan coba lagi nanti."})

    data = request.get_json()
    user_message = data.get('message', '').strip()

    try:
        
        
        # If no exact or similar match found, use QA pipeline
        relevant_contexts = find_most_relevant_context(user_message)
        print(f"Found {len(relevant_contexts)} relevant contexts")
        
        best_answer = ""
        best_score = 0
        best_context = ""
        
        for context in relevant_contexts:
            try:
                result = qa_pipeline(
                    question=user_message,
                    context=context,
                    handle_impossible_answer=True,
                    max_answer_len=100
                )

                print(f"QA Context: {context} ")
                print(f"QA Result: {result}")

                if result['score'] > best_score and result['answer'].strip():
                    best_answer = result['answer']
                    best_score = result['score']
                    best_context = context

                    if best_score > 0.8:
                        break
            except Exception as e:
                print(f"Error processing context: {e}")
                continue

        if best_answer and best_score > 0.3:
            return jsonify({
                'response': best_answer,
                'context': f"Konteks terkait: {best_context[:200]}..." if len(best_context) > 200 else best_context
            })
        else:
            sample_questions = random.sample(
                [pair['question'] for pair in QA_PAIRS], 
                min(3, len(QA_PAIRS)))
            return jsonify({
                'response': (
                    "Maaf, saya tidak menemukan jawaban yang tepat.\n"
                    "Contoh pertanyaan:\n\n" +
                    "\n".join(f"- {q}" for q in sample_questions) +
                    "\n\nSilakan coba pertanyaan lain."
                ),
                'context': "Tidak ditemukan konteks yang relevan"
            })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'response': "Maaf, terjadi kesalahan. Silakan coba pertanyaan lain.",
            'context': "Error occurred"
        })

if __name__ == '__main__':
    app.run(debug=True)