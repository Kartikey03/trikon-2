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

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it up.")

# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = api_key

def load_and_process_document(file_path):
    """
    Load a text document and split it into chunks.
    """
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
    
    print(f"Document loaded and split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks.
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created in memory")
        
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

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain QA System</title>
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
    </style>
</head>
<body>
    <h1>LangChain QA System</h1>
    <div class="container">
        <form action="/web-ask" method="post">
            <h3>Ask a Question:</h3>
            <textarea name="question" placeholder="Enter your question here..." required>{{ question }}</textarea>
            <button type="submit">Submit</button>
        </form>
        
        {% if answer %}
        <div class="answer">
            <strong>Answer:</strong>
            <p>{{ answer }}</p>
        </div>
        {% endif %}
        
        <div class="status">
            <p>System Status: {{ "Ready" if system_initialized else "Initializing..." }}</p>
        </div>
    </div>
</body>
</html>
"""

# Set up the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the QA system
qa_system = None

def initialize_qa_system(file_path):
    """Initialize the QA system with the specified document."""
    global qa_system
    print("Processing document and creating vector store. This may take a moment...")
    chunks = load_and_process_document(file_path)
    vector_store = create_vector_store(chunks)
    qa_system = create_qa_chain(vector_store)
    print("QA system initialized and ready to use")

@app.route('/', methods=['GET'])
def index():
    """Render the web interface."""
    return render_template_string(
        HTML_TEMPLATE, 
        system_initialized=qa_system is not None,
        question="",
        answer=None
    )

@app.route('/web-ask', methods=['POST'])
def web_ask():
    """Handle question submissions from the web interface."""
    if not qa_system:
        return render_template_string(
            HTML_TEMPLATE,
            system_initialized=False,
            question=request.form.get('question', ''),
            answer="System is still initializing. Please try again in a moment."
        )
    
    question = request.form.get('question', '')
    if not question:
        return redirect(url_for('index'))
    
    try:
        answer = qa_system({"query": question})
        return render_template_string(
            HTML_TEMPLATE,
            system_initialized=True,
            question=question,
            answer=answer["result"]
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            system_initialized=True,
            question=question,
            answer=f"Error: {str(e)}"
        )

@app.route('/ask', methods=['POST'])
def ask():
    """API endpoint for programmatic access."""
    if not qa_system:
        return jsonify({"error": "QA system not initialized"}), 500
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    try:
        answer = qa_system({"query": question})
        return jsonify({"answer": answer["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "system_initialized": qa_system is not None})

if __name__ == "__main__":
    # Configuration
    document_path = os.getenv("DOCUMENT_PATH", "your_document.txt")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    
    # Initialize the QA system before starting the server
    initialize_qa_system(document_path)
    
    # Start the Flask server
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=True)