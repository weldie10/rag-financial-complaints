# RAG Financial Complaints

A Retrieval-Augmented Generation (RAG) system for processing and analyzing financial complaints from the Consumer Financial Protection Bureau (CFPB).

## Project Structure

```
├── .vscode/
│   └── settings.json                # VS Code Python configuration
├── .github/
│   └── workflows/
│       └── unittests.yml            # GitHub Actions CI/CD
├── data/
│   ├── raw/                         # Original CFPB complaint dataset
│   └── processed/                   # Cleaned and filtered data
│       └── filtered_complaints.csv  # Preprocessed dataset
├── vector_store/                     # Persisted FAISS/ChromaDB index
├── notebooks/
│   ├── __init__.py
│   ├── README.md
│   ├── eda_and_preprocessing.ipynb      # Task 1: Interactive EDA notebook
│   └── chunking_embedding_indexing.ipynb # Task 2: Chunking and indexing notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data loading with error handling
│   ├── data_preprocessor.py             # Text cleaning and filtering
│   ├── eda.py                            # Exploratory data analysis
│   ├── preprocessing_pipeline.py         # Task 1: Complete preprocessing pipeline
│   ├── stratified_sampler.py             # Task 2: Stratified sampling
│   ├── text_chunker.py                   # Task 2: Text chunking strategies
│   ├── embedding_generator.py            # Task 2: Embedding generation
│   ├── vector_store_manager.py           # Task 2: FAISS/ChromaDB vector store
│   └── indexing_pipeline.py             # Task 2: Complete indexing pipeline
├── tests/
│   └── __init__.py
├── app.py                            # Gradio/Streamlit interface
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

3. Place the CFPB complaint dataset in `data/raw/complaints.csv`

## Usage

### Task 1: Exploratory Data Analysis and Preprocessing

#### Option 1: Run the Complete Pipeline (Script)
```bash
python src/preprocessing_pipeline.py
```

This will:
- Load the CFPB complaint dataset
- Perform EDA on raw data
- Filter by products (Credit card, Personal loan, Savings account, Money transfers)
- Clean text narratives
- Generate visualizations
- Save processed data to `data/processed/filtered_complaints.csv`

#### Option 2: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/eda_and_preprocessing.ipynb
```

The notebook provides an interactive environment for:
- Step-by-step EDA exploration
- Custom analysis and visualizations
- Experimenting with different preprocessing parameters

#### Using the Modules Programmatically

```python
from src.data_loader import load_cfpb_complaints
from src.data_preprocessor import preprocess_complaints, save_processed_data
from src.eda import EDAAnalysis

# Load data
df = load_cfpb_complaints()

# Perform EDA
eda = EDAAnalysis(df)
summary = eda.generate_summary_report()
print(summary)

# Preprocess data
df_processed = preprocess_complaints(df)

# Save processed data
save_processed_data(df_processed)
```

### Task 2: Text Chunking, Embedding, and Vector Store Indexing

#### Option 1: Run the Complete Indexing Pipeline (Script)
```bash
python src/indexing_pipeline.py
```

Or with custom parameters:
```bash
python src/indexing_pipeline.py \
    --sample-size 12000 \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
    --vector-store-type faiss
```

This will:
- Load processed complaint data
- Create stratified sample (10,000-15,000 complaints)
- Chunk text narratives using LangChain RecursiveCharacterTextSplitter
- Generate embeddings using sentence-transformers
- Create and save vector store (FAISS/ChromaDB)
- Store metadata with each vector for traceability

#### Option 2: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/chunking_embedding_indexing.ipynb
```

The notebook provides an interactive environment for:
- Experimenting with different chunk sizes and overlaps
- Testing different embedding models
- Analyzing chunking results
- Testing vector store search functionality

#### Using the Modules Programmatically

```python
from src.stratified_sampler import stratified_sample
from src.text_chunker import TextChunker
from src.embedding_generator import EmbeddingGenerator
from src.vector_store_manager import create_vector_store, save_vector_store

# Load processed data
df = pd.read_csv('data/processed/filtered_complaints.csv')

# Create stratified sample
df_sampled = stratified_sample(df, sample_size=12000, stratify_column='Product')

# Chunk texts
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
df_chunks = chunker.chunk_dataframe(df_sampled, text_column='Consumer complaint narrative')

# Generate embeddings
embedding_generator = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_generator.generate_embeddings_for_dataframe(df_chunks)

# Create and save vector store
vector_store = create_vector_store(store_type="faiss", embedding_dim=embeddings.shape[1])
vector_store.add_vectors(embeddings, metadata_list)
save_vector_store(vector_store, "vector_store/", store_type="faiss")
```

### Running the Application

Run the Gradio/Streamlit interface:
```bash
python app.py
```

## Module Documentation

### `src/data_loader.py`
- **`load_cfpb_complaints()`**: Loads the CFPB complaint dataset with error handling
- **`validate_dataframe()`**: Validates the loaded dataframe structure

### `src/data_preprocessor.py`
- **`preprocess_complaints()`**: Complete preprocessing pipeline
- **`filter_by_products()`**: Filter complaints by product type
- **`remove_empty_narratives()`**: Remove records with empty narratives
- **`clean_text()`**: Clean individual text narratives
- **`save_processed_data()`**: Save processed data to CSV

### `src/eda.py`
- **`EDAAnalysis`**: Class for performing exploratory data analysis
  - `get_basic_stats()`: Basic dataset statistics
  - `analyze_product_distribution()`: Product distribution analysis
  - `analyze_narrative_length()`: Word count statistics
  - `count_narratives()`: Narrative presence statistics
  - `plot_product_distribution()`: Visualize product distribution
  - `plot_narrative_length_distribution()`: Visualize narrative length distribution
  - `generate_summary_report()`: Generate text summary report

### `src/stratified_sampler.py`
- **`stratified_sample()`**: Create stratified sample maintaining proportional representation
- **`create_complaint_id()`**: Create or use existing complaint ID column
- **`get_sampling_statistics()`**: Compare original and sampled datasets
- **`save_sample()`**: Save sampled data to CSV

### `src/text_chunker.py`
- **`TextChunker`**: Class for chunking text narratives
  - `chunk_text()`: Chunk a single text
  - `chunk_dataframe()`: Chunk all texts in a dataframe with metadata preservation
- **`analyze_chunking_results()`**: Analyze and provide chunking statistics
- Supports LangChain RecursiveCharacterTextSplitter and custom chunking

### `src/embedding_generator.py`
- **`EmbeddingGenerator`**: Class for generating text embeddings
  - `generate_embeddings()`: Generate embeddings for a list of texts
  - `generate_embeddings_for_dataframe()`: Generate embeddings for dataframe texts
- **`get_model_info()`**: Get information about embedding models
- Default model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

### `src/vector_store_manager.py`
- **`FAISSVectorStore`**: FAISS-based vector store with metadata support
  - `add_vectors()`: Add vectors and metadata to index
  - `search()`: Search for similar vectors
  - `save()` / `load()`: Persist and load vector store
- **`ChromaDBVectorStore`**: ChromaDB-based vector store
- **`create_vector_store()`**: Factory function to create vector store
- **`save_vector_store()` / `load_vector_store()`**: Save and load utilities

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Features

- **Modular Design**: Reusable, well-structured modules for data loading, preprocessing, analysis, and indexing
- **Error Handling**: Comprehensive error handling with custom exceptions and logging
- **EDA Tools**: Automated exploratory data analysis with visualizations
- **Text Cleaning**: Advanced text preprocessing for improved embedding quality
- **Stratified Sampling**: Proportional sampling across product categories
- **Text Chunking**: Intelligent text chunking with LangChain support
- **Embedding Generation**: Fast embedding generation with sentence-transformers
- **Vector Store**: FAISS and ChromaDB support with metadata storage
- **CI/CD**: GitHub Actions workflow for automated testing

## Data Preprocessing Pipeline

The preprocessing pipeline includes:
1. **Data Loading**: Robust CSV loading with validation
2. **Product Filtering**: Filter to specific product categories
3. **Narrative Validation**: Remove empty or invalid narratives
4. **Text Cleaning**:
   - Lowercasing
   - Special character removal
   - Boilerplate text removal
   - Whitespace normalization

## Output

### Task 1: Preprocessing Pipeline
- **Processed Data**: `data/processed/filtered_complaints.csv`
- **Visualizations**: `data/processed/plots/` (product distribution, narrative length distribution)
- **Logs**: `preprocessing.log`

### Task 2: Indexing Pipeline
- **Sampled Data**: `data/processed/sampled_complaints.csv`
- **Vector Store**: `vector_store/` directory containing:
  - `faiss.index`: FAISS index file
  - `metadata.pkl`: Metadata for all vectors
  - `index_info.pkl`: Index configuration and statistics
- **Logs**: `indexing.log`

## Task 2 Report

For detailed documentation on Task 2 implementation, see the report section in the notebook `notebooks/chunking_embedding_indexing.ipynb` which covers:
- **Sampling Strategy**: Stratified sampling approach and justification
- **Chunking Approach**: Text chunking strategy, chunk size/overlap selection, and rationale
- **Embedding Model Choice**: Model selection, comparison with alternatives, and justification
- **Vector Store**: FAISS/ChromaDB implementation and metadata storage

## License

[Add your license here]

