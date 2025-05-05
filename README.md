# Time Series ChatTS Refactor

This project analyzes time series data using a large language model (LLM) and logs results.

## Structure
- **llm/**: LLM initialization and management
- **chat_history/**: Chat history persistence
- **data/**: Scenario loading and synthetic data generation
- **logs/**: Analysis logging
- **utils/**: Helper utilities (file management)
- **main.py**: Script entry point

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Update `model_path` in `main.py`.
3. Run: `python main.py`