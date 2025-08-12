# 🔺 Trix - AI-Powered Chatbot for Trikon 2.0

**Trix won't trick you!** 🤖

Trix is an intelligent AI chatbot designed as the official virtual assistant for Trikon 2.0 hackathon. Built using Retrieval Augmented Generation (RAG) architecture, Trix provides context-aware, personalized responses to help participants navigate the event seamlessly.

## 🚀 Features

- **RAG-Based AI**: Utilizes document retrieval and Google Gemini for accurate, context-aware responses
- **Event Assistant**: Helps with venue navigation, meal timings, hackathon rounds, and activities
- **Web Interface**: User-friendly Flask-based web application
- **API Endpoints**: RESTful API for programmatic access
- **Real-time Processing**: Efficient document chunking and vector similarity search
- **Friendly Persona**: Designed to communicate like talking to a 5-year-old for maximum accessibility

## 🛠️ Tech Stack

- **Backend**: Python, Flask, LangChain
- **AI/ML**: Google Gemini API, FAISS Vector Store
- **Document Processing**: RecursiveCharacterTextSplitter
- **Embeddings**: Google Generative AI Embeddings
- **Frontend**: HTML5, CSS3, JavaScript
- **Environment**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key (from [Google AI Studio](https://makersuite.google.com/app/apikey))
- Text document containing event information

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/trix-chatbot.git
   cd trix-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create requirements.txt**
   ```bash
   langchain
   langchain-google-genai
   google-generativeai
   faiss-cpu
   flask
   flask-cors
   python-dotenv
   ```

5. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   DOCUMENT_PATH=path/to/your/document.txt
   HOST=0.0.0.0
   PORT=5000
   ```

## 🚀 Usage

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Web Interface

- Navigate to the web interface
- Wait for system initialization (first time may take a few minutes)
- Ask questions about the event in the text area
- Get instant responses from Trix!

### API Usage

**Ask a question:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What time is lunch?"}'
```

**Health check:**
```bash
curl http://localhost:5000/health
```

## 📁 Project Structure

```
trix-chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md             # Project documentation
└── data/
    └── event_info.txt    # Your event document
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/ask` | POST | Ask questions programmatically |
| `/health` | GET | System health check |
| `/initialize` | POST | Initialize/restart the system |

## 🔧 Configuration

### Model Options

You can customize the Gemini model in `create_qa_chain()`:
- `gemini-1.5-flash` - Fast and efficient (default)
- `gemini-1.5-pro` - Higher quality responses

### Document Processing

Adjust text splitting parameters in `load_and_process_document()`:
- `chunk_size`: Size of each text chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

## 🎨 Customization

### Personality

Modify Trix's personality in the prompt template within `create_qa_chain()`. The current setup makes Trix:
- Fun and resourceful
- High humor
- Explains things simply (like talking to a 5-year-old)

### UI Styling

Update the `HTML_TEMPLATE` in `app.py` to customize the web interface appearance.

## 🔍 Troubleshooting

**Common Issues:**

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is correctly set in `.env`
2. **Document Not Found**: Check `DOCUMENT_PATH` points to your text file
3. **Initialization Timeout**: Large documents may take time to process
4. **Memory Issues**: Use `faiss-cpu` instead of `faiss-gpu` for CPU-only setups

**Debug Mode:**
```bash
export FLASK_ENV=development
python app.py
```

## 📊 Performance

- **Initialization**: ~2-5 minutes for medium documents
- **Query Response**: ~1-3 seconds
- **Memory Usage**: Depends on document size and chunk count
- **Concurrent Users**: Supports multiple simultaneous requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Trikon 2.0** - The hackathon that inspired this project
- **DevInt & Intellia** - Technical core team
- **Google Gemini** - AI language model
- **LangChain** - RAG framework
- **FAISS** - Vector similarity search

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Voice interaction capabilities  
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration
- [ ] Real-time event updates
- [ ] User feedback system

---

**"Trix won't trick you!"** - Built with ❤️ for Trikon 2.0
