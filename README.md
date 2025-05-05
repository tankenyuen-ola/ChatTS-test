# Time Series ChatTS Refactor

This project analyzes time series data using ChatTS-14B model and logs results.

## Structure
- **ChatTS/**: Original ChatTS implementation
- **llm/**: LLM initialization and management
- **chat_history/**: Chat history persistence
- **data/**: Scenario loading and synthetic data generation
- **logs/**: Analysis logging
- **utils/**: Helper utilities (file management)
- **prompt/**: Prompt Configuration
- **generated_data/**: Sample Generated Data
- **run_mode/**: Batch Analysis & Interactive Mode Implementation
- **main.py**: Script entry point

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Set up your LLM in `llm_api\llm`.
3. Run: `python main.py`

## Data Generation
1. Open `data\scenario_generator.py`
2. Add timeseries data in `CrisisDataSimulator()`, you may refer to the functions declared in the class.
3. Add the scenarios that required to be generated in `scenarios_to_run` variables.
4. Adjust the random seed to generate random data.
5. Run `data\scenario_generator.py` directly to generate the scenario data, the data will be saved in `generated_data/data`, `generated_data/meta` and `generated_data/plot` respectively.

## Notes
1. This repo is using vLLM for inference, please refer to original repo to download specific vLLM version `0.6.6.post1`.