
# AI Chatbot with PDF RAG

An AI-powered chatbot built using Django, Groq API, and Retrieval-Augmented Generation (RAG).

Users can upload PDF documents and ask questions about their content. The chatbot retrieves relevant information from the uploaded PDFs and uses an LLM to generate answers.

## Features

- AI chatbot powered by Groq API.
- Chat memory using Django sessions.
- Upload PDF documents.
- Extract text from PDF files.
- Split text into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store and search embeddings using ChromaDB.
- Retrieve relevant information using semantic search.
- Generate answers using RAG.

## Technologies Used

- Python
- Django
- Groq API
- Sentence Transformers
- ChromaDB
- PyMuPDF
- HTML
- CSS

## How It Works

1. User uploads a PDF document.
2. PyMuPDF extracts the text.
3. The extracted text is divided into chunks.
4. Sentence Transformers converts chunks into embeddings.
5. ChromaDB stores the embeddings.
6. When a user asks a question, semantic search retrieves relevant chunks.
7. Groq uses the retrieved information to generate an answer.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd ai-chatbot-website
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 7. Run database migrations

```bash
python manage.py migrate
```

### 8. Start the development server

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/ in your browser.

## Environment Variables

| Variable | Description |
|---|---|
| GROQ_API_KEY | Your Groq API key |

## Future Improvements

- Support multiple PDF documents.
- Improve document management.
- Improve retrieval accuracy.
- Add support for scanned PDFs.
- Improve the user interface.
