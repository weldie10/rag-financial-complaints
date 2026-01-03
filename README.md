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
│   └── eda_and_preprocessing.ipynb  # Interactive EDA notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading with error handling
│   ├── data_preprocessor.py         # Text cleaning and filtering
│   ├── eda.py                        # Exploratory data analysis
│   └── preprocessing_pipeline.py     # Complete preprocessing pipeline
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

- **Modular Design**: Reusable, well-structured modules for data loading, preprocessing, and analysis
- **Error Handling**: Comprehensive error handling with custom exceptions and logging
- **EDA Tools**: Automated exploratory data analysis with visualizations
- **Text Cleaning**: Advanced text preprocessing for improved embedding quality
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

After running the preprocessing pipeline:
- **Processed Data**: `data/processed/filtered_complaints.csv`
- **Visualizations**: `data/processed/plots/` (product distribution, narrative length distribution)
- **Logs**: `preprocessing.log`

## License

[Add your license here]

