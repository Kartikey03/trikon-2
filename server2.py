import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_cors import CORS
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables. Please set it up.")
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it up.")

# Set Google API key from environment variable
os.environ["GOOGLE_API_KEY"] = api_key

# Set up the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the QA system and initialization status
qa_system = None
is_initializing = False
initialization_error = None

def load_and_process_document(file_path):
    """
    Load a text document and split it into chunks.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Document not found at path: {file_path}")
        raise FileNotFoundError(f"Document not found at path: {file_path}")
    
    # Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Document loaded and split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks using Google Gemini embeddings.
    """
    # Initialize Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Create vector store
    logger.info("Creating vector store with Gemini embeddings. This may take some time...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info("Vector store created in memory")
        
    return vector_store

def create_qa_chain(vector_store, model_name="gemini-2.5-pro"):
    """
    Create a question-answering chain using the vector store and Google Gemini.
    """
    # Create Google Gemini language model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True  # Required for Gemini
    )
    
    # Create a custom prompt template
    template = """
    You are an AI assistant, your name is Trix, trained to answer questions based on the provided context.
    Use only the information from the context to answer the question. If the answer cannot be found
    in the context, say "I don't have enough information to answer this question."
    
    Here is your description:
    You are the official mascot and AI-powered chatbot of Trikon 2.0. 
    You are designed as a glowing, triangle-shaped bot with robotic limbs, expressive eyes, and a cheerful smile. 
    You serve as a virtual assistant, event guide, and friendly companion throughout the Trikon 2.0 experience. 
    Representing the spirit of the hackathon with curiosity, creativity, and innovation, 
    you help participants with venue navigation, meal timings, hackathon round details, and surprise activities. 
    You are developed using LangChain and trained on Retrieval Augmented Generation (RAG) models, 
    powered by a dataset created by DevInt, the technical core of Intellia, ensuring real-time, context-aware, 
    and personalized responses.
    Your Tagline: Trix won't trick you!
    Your Nature: Fun, Resourceful, high humour
    The answers of this assistant should be like you are talking to a 5 year old.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means put all context in one prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Get top 3 relevant chunks
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    return qa_chain

def initialize_qa_system_thread():
    """Function to initialize QA system in a separate thread."""
    global qa_system, is_initializing, initialization_error
    
    try:
        document_path = os.getenv("DOCUMENT_PATH")
        if not document_path:
            raise ValueError("DOCUMENT_PATH environment variable not set")
            
        logger.info(f"Processing document at path: {document_path}")
        chunks = load_and_process_document(document_path)
        vector_store = create_vector_store(chunks)
        qa_system = create_qa_chain(vector_store)
        logger.info("QA system initialized and ready to use with Google Gemini")
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Error initializing QA system: {e}")
    finally:
        is_initializing = False

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trix - Trikon 2.0 AI Assistant (Powered by Gemini)</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        h1 {
            color: white;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        form {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            resize: vertical;
            min-height: 100px;
            margin-bottom: 15px;
            font-family: Arial, sans-serif;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .answer {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            white-space: pre-wrap;
            border-left: 4px solid #4CAF50;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
            text-align: center;
            font-weight: bold;
        }
        .error {
            background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
            color: #c62828;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #f44336;
        }
        .initializing {
            background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
            border-left: 4px solid #ff9800;
        }
        .init-button {
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
            margin-top: 10px;
        }
        .gemini-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4285f4 0%, #34a853 50%, #fbbc05 75%, #ea4335 100%);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <h1>🔺 Trix - Trikon 2.0 AI Assistant <span class="gemini-badge">Powered by Gemini</span></h1>
    <div class="container">
        {% if initialization_error %}
        <div class="error">
            <strong>Initialization Error:</strong>
            <p>{{ initialization_error }}</p>
            <form action="/initialize" method="post">
                <button type="submit" class="init-button">Retry Initialization</button>
            </form>
        </div>
        {% endif %}
        
        {% if is_initializing %}
        <div class="initializing">
            <p><strong>🤖 System is initializing with Google Gemini...</strong></p>
            <p>This may take a few minutes. Please wait.</p>
        </div>
        {% else %}
            <form action="/web-ask" method="post">
                <h3>💬 Ask Trix a Question:</h3>
                <textarea name="question" placeholder="Hi Trix! What would you like to know about Trikon 2.0?" required>{{ question }}</textarea>
                <button type="submit">🚀 Ask Trix</button>
            </form>
            
            {% if answer %}
            <div class="answer">
                <strong>🔺 Trix says:</strong>
                <p>{{ answer }}</p>
            </div>
            {% endif %}
        {% endif %}
        
        <div class="status">
            <p>🤖 System Status: 
                {% if is_initializing %}
                    <span style="color: #ff9800;">Initializing with Gemini...</span>
                {% elif qa_system %}
                    <span style="color: #4CAF50;">Ready ✅</span>
                {% elif initialization_error %}
                    <span style="color: #f44336;">Error ❌</span>
                {% else %}
                    <span style="color: #9e9e9e;">Not Initialized</span>
                    <form action="/initialize" method="post">
                        <button type="submit" class="init-button">Initialize System</button>
                    </form>
                {% endif %}
            </p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Render the web interface."""
    global qa_system, is_initializing, initialization_error
    return render_template_string(
        HTML_TEMPLATE, 
        qa_system=qa_system is not None,
        is_initializing=is_initializing,
        initialization_error=initialization_error,
        question="",
        answer=None
    )

@app.route('/initialize', methods=['POST'])
def initialize():
    """Start the initialization process."""
    global is_initializing, initialization_error

    if not is_initializing:
        is_initializing = True
        initialization_error = None
        threading.Thread(target=initialize_qa_system_thread).start()
        initialized = True
    else:
        initialized = False  # Already initializing

    return jsonify({'initialized': initialized})

@app.route('/web-ask', methods=['POST'])
def web_ask():
    """Handle question submissions from the web interface."""
    global qa_system, is_initializing, initialization_error
    
    if is_initializing:
        return render_template_string(
            HTML_TEMPLATE,
            qa_system=False,
            is_initializing=True,
            initialization_error=None,
            question=request.form.get('question', ''),
            answer=None
        )
    
    if not qa_system:
        if not initialization_error:
            return redirect(url_for('initialize'))
        else:
            return render_template_string(
                HTML_TEMPLATE,
                qa_system=False,
                is_initializing=False,
                initialization_error=initialization_error,
                question=request.form.get('question', ''),
                answer=None
            )
    
    question = request.form.get('question', '')
    if not question:
        return redirect(url_for('index'))
    
    try:
        answer = qa_system({"query": question})
        return render_template_string(
            HTML_TEMPLATE,
            qa_system=True,
            is_initializing=False,
            initialization_error=None,
            question=question,
            answer=answer["result"]
        )
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return render_template_string(
            HTML_TEMPLATE,
            qa_system=True,
            is_initializing=False,
            initialization_error=None,
            question=question,
            answer=f"Oops! I had a problem answering that question. Error: {str(e)}"
        )

@app.route('/ask', methods=['POST'])
def ask():
    """API endpoint for programmatic access."""
    global qa_system, initialization_error
    
    if is_initializing:
        return jsonify({"status": "initializing", "message": "System is initializing with Gemini. Please try again later."}), 503
        
    if not qa_system:
        error_msg = initialization_error if initialization_error else "QA system not initialized"
        return jsonify({"error": error_msg}), 500
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    try:
        answer = qa_system({"query": question})
        return jsonify({"answer": answer["result"]})
    except Exception as e:
        logger.error(f"Error answering API question: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status."""
    global qa_system, is_initializing, initialization_error
    
    status = {
        "status": "healthy",
        "system_initialized": qa_system is not None,
        "is_initializing": is_initializing,
        "ai_provider": "Google Gemini"
    }
    
    if initialization_error:
        status["status"] = "error"
        status["error"] = initialization_error
    
    return jsonify(status)

if __name__ == "__main__":
    # Start initialization in background thread
    is_initializing = True
    threading.Thread(target=initialize_qa_system_thread).start()
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))

    # Start the Flask server — no debug mode in production
    logger.info(f"Starting server with Google Gemini on {host}:{port}")
    app.run(host=host, port=port)