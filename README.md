# RAG Financial Complaints

A Retrieval-Augmented Generation (RAG) system for processing and analyzing financial complaints.

## Project Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                       
│   └── processed/
├── vector_store/                   # Persisted FAISS/ChromaDB index
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── app.py                          # Gradio/Streamlit interface
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app.py
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

## License

[Add your license here]

