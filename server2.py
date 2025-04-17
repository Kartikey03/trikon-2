import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
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
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables. Please set it up.")
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it up.")

# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = api_key

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
    Create a FAISS vector store from document chunks.
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    logger.info("Creating vector store. This may take some time...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info("Vector store created in memory")
        
    return vector_store

def create_qa_chain(vector_store, model_name="gpt-3.5-turbo"):
    """
    Create a question-answering chain using the vector store.
    """
    # Create language model
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    # Create a custom prompt template
    template = """
    You are an AI assistant trained to answer questions based on the provided context.
    Use only the information from the context to answer the question. If the answer cannot be found
    in the context, say "I don't have enough information to answer this question.
    Description of You:
    Trix is the official mascot of Trikon 2.0, designed as a glowing, triangle-shaped AI bot with big eyes, 
    a bright smile, and robotic limbs. Friendly, witty, and curious, 
    Trix acts as a virtual host and guide throughout the hackathon. 
    It functions as an interactive chatbot, helping with schedules, registration, venue navigation, and updates. 
    Trix also boosts engagement through providing  a chatbot . 
    Representing AI and automation, Trix symbolizes the core spirit of Trikon 2.0—innovation, creativity, 
    and tech-forward thinking—while also serving as a branding icon for sponsors and enhancing 
    the overall participant experience.
    Trix Tagline: Trix won't trick you
    Nature of assistant: Fun, Resourful, high humour
    The answers of this assistant should be like it is talking to a 5 year old."
    
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
        logger.info("QA system initialized and ready to use")
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
    <title>Trix - Trikon 2.0 AI Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        form {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            min-height: 100px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .answer {
            background-color: #e9f7ef;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
            white-space: pre-wrap;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            background-color: #e3f2fd;
            text-align: center;
        }
        .error {
            background-color: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }
        .initializing {
            background-color: #fff9c4;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
            text-align: center;
        }
        .init-button {
            background-color: #2196F3;
            margin-top: 10px;
        }
        .init-button:hover {
            background-color: #0b7dda;
        }
    </style>
</head>
<body>
    <h1>Trix - Trikon 2.0 AI Assistant</h1>
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
            <p><strong>System is initializing...</strong> This may take a few minutes. Please wait.</p>
        </div>
        {% else %}
            <form action="/web-ask" method="post">
                <h3>Ask Trix a Question:</h3>
                <textarea name="question" placeholder="Enter your question here..." required>{{ question }}</textarea>
                <button type="submit">Ask</button>
            </form>
            
            {% if answer %}
            <div class="answer">
                <strong>Trix says:</strong>
                <p>{{ answer }}</p>
            </div>
            {% endif %}
        {% endif %}
        
        <div class="status">
            <p>System Status: 
                {% if is_initializing %}
                    Initializing...
                {% elif qa_system %}
                    Ready
                {% elif initialization_error %}
                    Error
                {% else %}
                    Not Initialized
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
        return jsonify({"status": "initializing", "message": "System is initializing. Please try again later."}), 503
        
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
        "is_initializing": is_initializing
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
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port)