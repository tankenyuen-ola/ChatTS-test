from chat_history.history import update_chat_history, save_chat_history
from llm_api.load import load_model
from llm_api.manager import release_llm
from logs.save import save_log
import numpy as np

def batch_analyse(scenarios, llm, tokenizer, processor, log_file):
    if scenarios:
        print("Loaded data structure:")
        # Example of accessing data:
        for scenario_key, scenario_info in scenarios.items():
            print(f"\nScenario: {scenario_key}")
            print(f"  Title: {scenario_info['title']}")
            print(f"  Metadata: {scenario_info['metadata_description']}")
            print(f"  Time Series Keys: {list(scenario_info['data'].keys())}")
            # Example: Accessing a specific time series array
            # if 'Build Duration (sec)' in scenario_info['data']:
            #    build_duration_ts = scenario_info['data']['Build Duration (sec)']
            #    print(f"  Build Duration data length: {len(build_duration_ts)}")
    else:
        print("Failed to load data. Exiting.")
        exit() # Exit if data loading failed
        
    # --- Main Analysis Loop ---
    for scenario_key, scenario_info in scenarios.items():
        print(f"\nAnalyzing Scenario: {scenario_info['title']}")
        
        try:
            # --- Prepare all timeseries data for the current scenario ---
            all_timeseries_data = {}
            valid_scenario = True
            for ts_name, timeseries in scenario_info['data'].items():
                # Ensure timeseries is a numpy array
                if not isinstance(timeseries, np.ndarray):
                    print(f"    Skipping {ts_name} in scenario {scenario_info['title']}: Data is not a numpy array.")
                    # Decide if you want to skip the whole scenario or just this series
                    # valid_scenario = False 
                    # break # Option 1: Skip entire scenario if one series is bad
                    continue # Option 2: Skip just this series

                # Convert to float if necessary
                try:
                    all_timeseries_data[ts_name] = timeseries.astype(float)
                except ValueError as ve:
                    print(f"    Skipping {ts_name} in scenario {scenario_info['title']}: Cannot convert data to float. Error: {ve}")
                    # Decide handling: skip series or scenario
                    continue 

            # If you chose to skip the scenario on bad data, check the flag
            # if not valid_scenario:
            #     continue

            if not all_timeseries_data:
                print(f"Skipping scenario {scenario_info['title']}: No valid time series data found.")
                continue

            # --- Call the analysis function with all timeseries for the scenario ---
            title = scenario_info['title']
            indicators = list(all_timeseries_data.keys()) # Use keys from the processed data

            # NOTE: You might need to adapt 'load_model' to accept a dictionary 
            # of timeseries instead of a single numpy array.
            # The current 'load_model' signature expects a single 'timeseries' argument.
            # You'll need to decide how to pass 'all_timeseries_data' to it.
            # Example: Pass the dictionary directly if load_model is updated.
            response = load_model(all_timeseries_data, title, indicators, llm, tokenizer, processor, None, None) 

            # --- Log the analysis for the entire scenario ---
            # NOTE: You might need to adapt 'analyze_log' as well.
            # It currently expects a single timeseries list and name.
            # Example: Pass the dictionary of lists and the scenario title.
            all_timeseries_lists = {name: ts.tolist() for name, ts in all_timeseries_data.items()}
            save_log(log_file, all_timeseries_lists, title, response) # Pass dict and title
        except Exception as e:
            print(f"Error processing scenario {scenario_info['title']}: {e}")

def interactive_chat_ts(scenarios, llm, tokenizer, processor):
    """
    Interactive Q&A over a chosen scenario with persistent chat history.
    """

    print("Available scenarios:")
    for key, info in scenarios.items():
        print(f"  {key} → {info['title']}")
    choice = input("Enter scenario key (or 'exit'): ").strip()
    if choice.lower() in ('exit','quit'):
        return
    if choice not in scenarios:
        print(f"❌ Unknown key: {choice}")
        return

    # 3. Prepare scenario data
    info = scenarios[choice]
    ts_data = info['data']
    title = info['title']
    indicators = list(ts_data.keys())

    # 4. Initialize model/tokenizer/processor
    print(f"✅ Model initialized for '{choice}' ({title}).")

    # 5. Load or init chat history
    chat_history = []

    try:
        while True:
            # User input
            user_q = input("\nYou ▶ ").strip()
            if user_q.lower() in ('exit','quit'):
                break

            chat_history = update_chat_history(chat_history, user_q, None)

            # 7. Delegate to load_model (injecting question and history)
            response = load_model(
                all_timeseries_data=ts_data,
                title=title,
                indicators=indicators,
                model=llm,
                tokenizer=tokenizer,
                processor=processor,
                question=user_q,
                chat_history=chat_history
            )

            # 8. Print and record model response
            print(f"\nAssistant ▶ {response.strip()}")
            chat_history = update_chat_history(chat_history, user_q, response)
    finally:
        # 9. Persist history
        save_chat_history(chat_history)