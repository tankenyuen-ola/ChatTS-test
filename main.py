import os
from llm_api.manager import initialize_llm, release_llm
from chat_history.history import update_chat_history, save_chat_history
from logs.save import save_log
from data.scenario_loader import load_scenarios
from utils.file_utils import file_duplicate
from run_mode.modes import batch_analyse, interactive_chat_ts

global llm
global tokenizer
global processor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'log')
    pic_dir = os.path.join(base_dir, 'pic')

    log_file = os.path.join(base_dir, 'logs', 'timeseries_analysis.txt')
    log_file = file_duplicate(log_file)

    data_dir = os.path.join(os.path.join(base_dir,"generated_data"))
    print(data_dir)
    model_dir = os.path.join(os.path.join(base_dir,"llm_api/llm"))

    scenarios = load_scenarios(data_dir)
    if scenarios is None:
        # No valid .npz/.txt found ⇒ fall back to synthetic data
        print("No scenario files detected – generating synthetic series instead.")
        exit()

    try:
        llm, tokenizer, processor = initialize_llm(model_dir)
        if llm is None or tokenizer is None or processor is None:
            print("Failed to initialize LLM. Exiting.")
            exit()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        exit()
    chat_history = []

    mode = input("Run in (1) batch-analyze or (2) interactive mode? [1/2]: ").strip()
    if mode == "1":
        batch_analyse(scenarios, llm, tokenizer, processor, log_file)
    elif mode == "2":
        interactive_chat_ts(scenarios, llm, tokenizer, processor)

    # for key, info in scenarios.items():
    #     data = info['data']
    #     title = info.get('Title', key)
    #     # Prepare prompt and call llm
    #     #response = llm.process(data)  # placeholder
    #     save_log(log_file, {title: data.tolist()}, title, response)
    #     chat_history = update_chat_history(chat_history, title, response)
    #     save_chat_history(chat_history)

    release_llm()
    print("✅ Model resources released.")
    #generate_deprecated_series(os.path.join(base_dir, 'pic'))

if __name__ == '__main__':
    main()