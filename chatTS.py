from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sys

import ChatTS.chatts.vllm.chatts_vllm
from vllm import LLM, SamplingParams

# Global LLM instance that will be reused
_global_llm_instance = None
_global_tokenizer_instance = None
_global_processor_instance = None

def initialize_llm(model_path):
    """Initialize and return a persistent LLM instance"""
    global _global_llm_instance
    global _global_tokenizer_instance
    global _global_processor_instance
    
    if (_global_llm_instance is not None or _global_tokenizer_instance is not None or _global_processor_instance is not None):
        return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance
    
    try:
        # _global_llm_instance = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype='float16', local_files_only=True, trust_remote_code=True)
        _global_llm_instance = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=2, gpu_memory_utilization=0.9, max_model_len=8192, limit_mm_per_prompt={"timeseries": 50}, max_num_seqs=128, dtype='half')
        _global_tokenizer_instance = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        _global_processor_instance = AutoProcessor.from_pretrained(model_path, tokenizer=_global_tokenizer_instance, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error during LLM initialization: {e}")
        # Reset globals if initialization failed partially
        _global_llm_instance = None
        _global_tokenizer_instance = None
        _global_processor_instance = None
        return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

    return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

def get_llm_instance():
    """Get the current LLM instance if initialized"""
    global _global_llm_instance
    global _global_tokenizer_instance
    global _global_processor_instance
    return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

def release_llm():
    """Release the LLM resources when done"""
    global _global_llm_instance
    global _global_tokenizer_instance
    global _global_processor_instance
    _global_llm_instance = None
    _global_tokenizer_instance = None
    _global_processor_instance = None
    # Additional cleanup could go here if needed

def update_chat_history(chat_history, user_message, model_response):
    if model_response is None:
        chat_history.append(
            {
                "role": "user", "content": user_message
            }
        )
    else:
        chat_history.append(
            {
                "role": "assistant", "content": model_response
            }
        ) 
    
    return chat_history

def save_chat_history(chat_history):
    """
    将聊天历史保存到文件
    
    Args:
        chat_history: 聊天历史对象
    """
    
    # 保存到文件
    with open(os.path.join(os.path.dirname(__file__), "chat_history", "chat_history.json"), "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)   
    print("聊天历史已保存到 chat_history.json 文件")

def load_model(all_timeseries_data, title, indicators, model, tokenizer, processor, question=None, chat_history=None):
    """
    Generates an analysis response from the LLM for a scenario with multiple time series.

    Args:
        all_timeseries_data (dict): Dictionary where keys are indicator names (str)
                                     and values are the corresponding time series (np.ndarray).
        title (str): The title of the scenario.
        indicators (list): A list of indicator names (str) corresponding to the order
                           of timeseries expected by the processor.
        model: The loaded Hugging Face model instance.
        tokenizer: The loaded Hugging Face tokenizer instance.
        processor: The loaded Hugging Face processor instance.

    Returns:
        str: The generated analysis response from the LLM, or an error message.
    """
    # Create time series and prompts
    indicator_string =', '.join(indicators) # Join the list of indicator names

    # Prepare the list of timeseries for the processor IN THE ORDER SPECIFIED BY 'indicators'
    timeseries_list_for_processor = []
    valid_indicators_processed = [] # Keep track of indicators we actually add
    ts_lengths = []
    #print(all_timeseries_data)

    for indicator in indicators: # Iterate in the order provided by the list
        if indicator in all_timeseries_data:
            #print(indicator)
            ts_data = all_timeseries_data[indicator]
            # Basic validation
            if isinstance(ts_data, np.ndarray) and ts_data.ndim == 1 and len(ts_data) > 0:
                 # Convert to list of floats/ints for the processor
                 try:
                      timeseries_list_for_processor.append(ts_data.astype(float).tolist())
                      ts_lengths.append(len(ts_data))
                      valid_indicators_processed.append(indicator) # Add indicator name if data is valid
                 except ValueError:
                      print(f"Warning: Could not convert timeseries '{indicator}' to float list. Skipping.")
            elif not isinstance(ts_data, np.ndarray):
                 print(f"Warning: Data for indicator '{indicator}' is not a numpy array. Skipping.")
            elif ts_data.ndim != 1:
                 print(f"Warning: Data for indicator '{indicator}' is not 1-dimensional. Skipping.")
            else: # len(ts_data) == 0
                 print(f"Warning: Data for indicator '{indicator}' is empty. Skipping.")
        else:
             print(f"Warning: Indicator '{indicator}' not found in all_timeseries_data dictionary. Skipping.")

    if not timeseries_list_for_processor:
        return "Error: No valid time series data provided to analyze."

    # --- Correction Start ---
    # Generate the correct number of <ts><ts/> placeholders
    num_valid_series = len(timeseries_list_for_processor)
    ts_placeholders = " ".join(["<ts><ts/>"] * num_valid_series) # Create one placeholder per valid series

    # Update the indicator string to only include indicators that were successfully processed
    valid_indicator_string = ', '.join(valid_indicators_processed)

    # Construct the prompt using the collected information and dynamic placeholders
    # Adjusted prompt to reflect multiple series and placeholders
    if question is None:
        prompt = (
            f"I have {num_valid_series} time series datasets related to '{title}', with the following indicators: {valid_indicator_string} respectively. "
            f"The data series are provided here: {ts_placeholders}. "
            f"Please perform a comprehensive analysis that includes:\n"
            f"1. Identifying local and global patterns, trends, and anomalies within each time series.\n"
            f"2. Exploring correlations and potential causal relationships between the different indicators.\n"
            f"3. Linking observed changes to plausible real-world scenarios or events, better related to {title}, that could have influenced these patterns.\n"
            f"4. Excluding potential causes that are inconsistent with the observed data trends.\n"
            f"Provide a detailed explanation of your findings, including any assumptions made and the reasoning behind your conclusions."
        )
    else:
        prompt = (
            f"I have {num_valid_series} time series datasets related to '{title}', with the following indicators: {valid_indicator_string} respectively. "
            f"The data series are provided here: {ts_placeholders}. "
            f"Here is the chat history between the user and the assistant that may help to answer the following question better. \n"
            f"{chat_history[:-1]} \n"
            f"Here is the question that is asked by user: {question}. Please answer the question in detail based on the given data, and you may refer to the chat history if needed. \n"
        )
    # --- Correction End ---

    print(f"Generated Prompt:\n{prompt}") # Debugging output

    # Apply Chat Template
    chat_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" # Use \n for clarity

    # Convert to tensor using the list of timeseries lists
    try:
        # Ensure the processor receives the timeseries in the same order as the placeholders were generated
        mm_data = {"timeseries": timeseries_list_for_processor}
        inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data
        }
        # inputs = processor(text=[chat_prompt], timeseries=timeseries_list_for_processor, padding=True, return_tensors="pt")

        # Move inputs to the same device as the model
        #inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #device = torch.device("cuda:0")
        #inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"Input processed.") # Debugging output

    except Exception as e:
        print(f"Error during processor step: {e}")
        # Log sample data if helpful
        print(f"Number of placeholders: {num_valid_series}")
        print(f"Number of timeseries lists passed: {len(timeseries_list_for_processor)}")
        exit()
        # print(f"Timeseries list sample: {[ts[:10] for ts in timeseries_list_for_processor]}") # Uncomment for more detail
        #return f"Error processing input: {e}"

    # Model Generate using the passed model instance
    try:
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(inputs, sampling_params=SamplingParams(max_tokens=3000)) # Increased max_new_tokens for potentially longer analysis
            # Inference
            #outputs = model.generate(inputs, sampling_params=SamplingParams(max_tokens=300))

            # for o in outputs:
            #     generated_text = o.outputs[0].text
            #     print(generated_text)
            # print("Model output generated.") # Debugging output
            #print(outputs[0].outputs[0].text)
            response = outputs[0].outputs[0].text
            response = response.removeprefix('[]\n\n')
            return response
    except Exception as e:
        print(f"Error during model generation: {e}")
        return f"Error generating response: {e}"

    #Decode response using the passed tokenizer instance
    #Ensure input_ids exists before accessing shape
    # if 'input_ids' in inputs and inputs['input_ids'] is not None:
    #     input_ids_len = inputs['input_ids'].shape[1] # Get length from the tensor
    #     if outputs[0].shape[0] > input_ids_len:
    #         response = tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True)
    #         print("Decode done.")
    #     else:
    #         # If output is not longer, decode the whole thing but warn
    #         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         print("Warning: Model output length was not greater than input length. Decoding full output.")
    #         print("Decode done.")
    # else:
    #     print("Error: 'input_ids' not found in processor output. Cannot decode properly.")
    #     response = "Error decoding response: input_ids missing."

def file_duplicate(filepath):
    # Handle duplicate files
    counter = 1
    
    # Split the filename and extension
    filename, extension = os.path.splitext(filepath)
    if not extension:  # If no extension provided, default to .txt
        extension = ".txt"
        filepath = filename + extension
    
    # Check if file exists and generate a new name if it does
    while os.path.exists(filepath):
        filepath = f"{filename}_{counter}{extension}"
        counter += 1

    return filepath

def analyze_log(filepath, timeseries_data, scenario_title, response):
    """
    Logs the analysis for a scenario with multiple time series to a text file.

    Args:
        filepath (str): Path to the text file.
        timeseries_data (dict): Dictionary of timeseries lists {name: list}.
        scenario_title (str): The title of the scenario being logged.
        response (str): The analysis response from the LLM.
    """
    with open(filepath, 'a') as file:
        file.write(f"--- Scenario Analysis: '{scenario_title}' ---\n")
        # Optionally log the timeseries data (can be very long)
        # for name, data in timeseries_data.items():
        #     file.write(f"Timeseries '{name}': {data[:20]}... (first 20 points)\n") # Log snippet
        file.write(f"Response:\n{response}\n")
        file.write("-" * (len(scenario_title) + 24) + "\n\n") # Separator

def deprecated_data():
    ### Seasonal Time Series (Annual Cycle) ###
    # Time vector: 365 days
    days = np.arange(365)

    # Simulate daily temperature with annual seasonality
    seasonal = 10 + 5 * np.sin(2 * np.pi * days / 365)  # One cycle per year
    noise = np.random.normal(scale=1.0, size=365)
    seasonal_series = seasonal + noise
    timeseries_data.append(seasonal_series)
    timeseries_name.append("Seasonal Time Series (Annual Cycle)")

    # Plot the seasonal time series
    plt.figure(figsize=(10, 4))
    plt.plot(days, seasonal_series, label='Seasonal Pattern')
    plt.title('Seasonal Time Series (Annual Cycle)')
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "seasonal_time_series.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory


    ### Cyclical Time Series (Economic Cycle) ###
    # Time vector: 200 months
    months = np.arange(200)

    # Simulate economic cycles with varying periods
    cycle = 2 * np.sin(2 * np.pi * months / 60)  # Approximate 5-year cycle
    noise = np.random.normal(scale=0.5, size=200)
    cyclical_series = cycle + noise
    timeseries_data.append(cyclical_series)
    timeseries_name.append("Cyclical Time Series (Economic Cycle)")

    # Plot the cyclical time series
    plt.figure(figsize=(10, 4))
    plt.plot(months, cyclical_series, label='Cyclical Pattern')
    plt.title('Cyclical Time Series (Economic Cycle)')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "cyclical_time_series.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ### Combined Time Series: Trend + Seasonality + Cycle
    # Time vector: 200 months
    months = np.arange(200)

    # Components
    trend = 0.05 * months  # Linear upward trend
    seasonal = 1.5 * np.sin(2 * np.pi * months / 12)  # Annual seasonality
    cycle = 2 * np.sin(2 * np.pi * months / 60)  # Approximate 5-year cycle
    noise = np.random.normal(scale=0.5, size=200)

    # Combined time series
    combined_series = trend + seasonal + cycle + noise
    timeseries_data.append(combined_series)
    timeseries_name.append("Combined Time Series: Trend + Seasonality + Cycle")
    # Plot the combined time series
    plt.figure(figsize=(10, 4))
    plt.plot(months, combined_series, label='Combined Pattern')
    plt.title('Combined Time Series: Trend + Seasonality + Cycle')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "combined_time_series.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ### Structural Breaks
    np.random.seed(0)
    n = 300
    x = np.arange(n)
    y = np.concatenate([
        np.random.normal(loc=5, scale=1, size=100),
        np.random.normal(loc=10, scale=1, size=100),
        np.random.normal(loc=7, scale=1, size=100)
    ])
    timeseries_data.append(y)
    timeseries_name.append("Structural Breaks")
    # Plot the combined time series
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label='Structural Breaks')
    plt.axvline(100, color='red', linestyle='--', label='Break Point')
    plt.axvline(200, color='red', linestyle='--')
    plt.title('Time Series with Structural Breaks')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "structural_breaks.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ### Mean Reversion
    np.random.seed(1)
    n = 300
    x = np.arange(n)
    y = np.zeros(n)
    mean = 10
    alpha = 0.9
    noise = np.random.normal(0, 1, n)

    for t in range(1, n):
        y[t] = alpha * y[t-1] + (1 - alpha) * mean + noise[t]

    timeseries_data.append(y)
    timeseries_name.append("Mean Reversion")
    # Plot the combined time series
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label='Mean Reversion')
    plt.axhline(mean, color='red', linestyle='--', label='Mean')
    plt.title('Mean Reverting Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "mean_reversion.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ### Intermittency
    np.random.seed(2)
    n = 300
    x = np.arange(n)
    y = np.zeros(n)

    for t in range(n):
        if np.random.rand() < 0.1:
            y[t] = np.random.randint(1, 10)
    
    timeseries_data.append(y)
    timeseries_name.append("Intermittency")
    plt.figure(figsize=(10, 3))
    plt.plot(x, y, drawstyle='steps-post', label='Intermittent Demand')
    plt.title('Intermittent Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "intermittency.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ### Level Shifts
    np.random.seed(3)
    n = 300
    x = np.arange(n)
    y = np.random.normal(loc=5, scale=1, size=n)
    y[150:] += 5  # Level shift at t=150
    timeseries_data.append(y)
    timeseries_name.append("Level Shifts")
    plt.figure(figsize=(10, 3))
    plt.plot(x, y, label='Level Shift')
    plt.axvline(150, color='red', linestyle='--', label='Shift Point')
    plt.title('Time Series with Level Shift')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "level_shifts.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ### Memory Usage Over Time ###
    np.random.seed(42)
    n = 300
    time = np.arange(n)

    # Simulate gradual memory increase with random noise
    memory_usage = np.cumsum(np.random.normal(loc=0.5, scale=0.2, size=n))

    # Introduce sudden drops to mimic garbage collection or restarts
    drop_indices = np.random.choice(n, size=5, replace=False)
    for idx in drop_indices:
        memory_usage[idx:] -= np.random.uniform(10, 20)

    # Ensure memory usage doesn't go below zero
    memory_usage = np.clip(memory_usage, a_min=0, a_max=None)
    timeseries_data.append(memory_usage)
    timeseries_name.append("Memory Usage Over Time")
    plt.figure(figsize=(10, 4))
    plt.plot(time, memory_usage, label='Memory Usage')
    plt.title('Simulated Memory Usage Fluctuations')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "memory_usage_over_time.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    ## Network Traffic Over Time ###
    np.random.seed(24)
    n = 300
    time = np.arange(n)

    # Simulate baseline network traffic with random noise
    network_traffic = np.random.normal(loc=50, scale=5, size=n)

    # Introduce random spikes
    spike_indices = np.random.choice(n, size=10, replace=False)
    for idx in spike_indices:
        network_traffic[idx] += np.random.uniform(50, 100)
    # Ensure network traffic doesn't go below zero
    network_traffic = np.clip(network_traffic, a_min=0, a_max=None)
    timeseries_data.append(network_traffic)
    timeseries_name.append("Network Traffic Over Time")
    plt.figure(figsize=(10, 4))
    plt.plot(time, network_traffic, label='Network Traffic')
    plt.title('Simulated Network Traffic with Spikes')
    plt.xlabel('Time')
    plt.ylabel('Traffic (Mbps)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot instead of showing it
    new_pic_path = file_duplicate(os.path.join(pic_path, "network_traffic_over_time.png"))
    plt.savefig(new_pic_path, dpi=300)
    plt.close()  # Close the figure to free up memory

def load_scenario_data(data_directory):
    """
    Loads time series data and metadata from all matching npz/txt pairs in a directory.

    Args:
        data_directory (str): Path to the directory containing .npz and .txt files.

    Returns:
        dict: A dictionary where keys are scenario base names (e.g., 'cicd')
              and values are dictionaries containing 'title', 'metadata_description',
              and 'data' (the time series arrays from the npz file).
              Returns an empty dictionary if the directory is not found or no matching pairs exist.
    """
    all_scenario_data = {}

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory not found at {data_directory}")
        return all_scenario_data

    # Find all npz files in the directory
    npz_dir = os.path.join(data_directory, 'data')
    meta_dir = os.path.join(data_directory, 'meta')
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('_data.npz')] # Assuming '_data.npz' suffix

    for npz_filename in npz_files:
        base_key = npz_filename.replace('_data.npz', '')
        npz_path = os.path.join(npz_dir, npz_filename)
        txt_filename = f"{base_key}_meta.txt" # Assuming '_meta.txt' suffix
        txt_path = os.path.join(meta_dir, txt_filename)

        scenario_data = {}
        time_series_arrays = {}
        title = f"Unknown Title ({base_key})" # Default title includes base key
        metadata_keys_line = f"Unknown Metadata ({base_key})" # Default metadata includes base key

        # Load data from NPZ file
        try:
            with np.load(npz_path) as data:
                time_series_arrays = {key: data[key] for key in data.files}
        except FileNotFoundError:
            print(f"Error: NPZ file not found at {npz_path} (Skipping scenario {base_key})")
            continue # Skip this scenario if NPZ is missing
        except Exception as e:
            print(f"Error loading NPZ file {npz_path}: {e} (Skipping scenario {base_key})")
            continue # Skip this scenario on NPZ load error

        # Load metadata from TXT file (optional)
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    found_title = False
                    found_meta = False
                    for line in lines:
                        if line.startswith("Title:"):
                            title = line.split(":", 1)[1].strip()
                            found_title = True
                        elif line.startswith("Scenario Metadata:"):
                            metadata_keys_line = line.strip()
                            found_meta = True
                    if not found_title:
                         print(f"Warning: 'Title:' not found in {txt_path}")
                    if not found_meta:
                         print(f"Warning: 'Scenario Metadata:' not found in {txt_path}")

            except Exception as e:
                print(f"Error reading TXT file {txt_path}: {e}. Metadata might be incomplete.")
        else:
            print(f"Warning: TXT file not found at {txt_path}. Metadata will be missing for scenario {base_key}.")

        # Store the loaded data for this scenario
        all_scenario_data[base_key] = {
            'title': title,
            'metadata_description': metadata_keys_line,
            'data': time_series_arrays
        }
        print(f"Successfully loaded scenario: {base_key}")

    if not all_scenario_data:
        print(f"No matching '_data.npz' and '_meta.txt' pairs found in {data_directory}")

    return all_scenario_data

def interactive_chat_ts(data_dir, model_path):
    """
    Interactive Q&A over a chosen scenario with persistent chat history.
    """
    # 1. Load all scenarios
    scenarios = load_scenario_data(data_dir)
    if not scenarios:
        print(f"No scenarios found in {data_dir}.")
        return

    # 2. Choose scenario
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
    try:
        model, tokenizer, processor = initialize_llm(model_path)
        if model is None:
            print("Failed to initialize LLM. Exiting.")
            exit()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        exit()
    print(f"✅ Model initialized for '{choice}' ({title}).")

    # 5. Load or init chat history
    chat_history = []

    try:
        while True:
            # User input
            user_q = input("\nYou ▶ ").strip()
            if user_q.lower() in ('exit','quit'):
                break

            # 6. Update history with user question (model_response empty)
            chat_history = update_chat_history(chat_history, user_q, None)

            # 7. Delegate to load_model (injecting question and history)
            response = load_model(
                all_timeseries_data=ts_data,
                title=title,
                indicators=indicators,
                model=model,
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

        # 10. Cleanup
        release_llm()
        print("✅ Model resources released.")

if __name__ == "__main__":
    file_path = os.path.abspath(__file__) # Changed base path
    base_output_dir = os.path.dirname(file_path)
    # timeseries_data = []
    # timeseries_name = []
        
    # Define output paths
    log_dir = os.path.join(base_output_dir, 'log')
    pic_dir = os.path.join(base_output_dir, 'pic')
    
    # Create output directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pic_dir, exist_ok=True)

    log_path = os.path.join(base_output_dir,"log/timeseries_analyze.txt")
    log_path = file_duplicate(log_path)
    pic_path = os.path.join(os.path.join(base_output_dir,"pic"))

    # Define input data paths
    data_dir = os.path.join(os.path.join(base_output_dir,"generated_data"))
    model_dir = os.path.join(os.path.join(base_output_dir,"ChatTS/ckpt"))

    mode = input("Run in (1) batch-analyze or (2) interactive mode? [1/2]: ").strip()
    if mode == "2":
        interactive_chat_ts(data_dir, model_dir)
    else:
        # … your existing batch loop …

        # --- Load Data ---
        combined_data = load_scenario_data(data_dir)

        if combined_data:
            print("Loaded data structure:")
            # Example of accessing data:
            for scenario_key, scenario_info in combined_data.items():
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
        
        ## --- Initialize Model (Optional - Uncomment if you want to run analysis) ---
        print(f"Initializing model from: {model_dir}")
        try:
            model, tokenizer, processor = initialize_llm(model_dir)
            if model is None:
                print("Failed to initialize LLM. Exiting.")
                exit()
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            exit()

        # --- Main Analysis Loop ---
        for scenario_key, scenario_info in combined_data.items():
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
                    print(f"    Skipping scenario {scenario_info['title']}: No valid time series data found.")
                    continue

                # --- Call the analysis function with all timeseries for the scenario ---
                title = scenario_info['title']
                indicators = list(all_timeseries_data.keys()) # Use keys from the processed data

                # NOTE: You might need to adapt 'load_model' to accept a dictionary 
                # of timeseries instead of a single numpy array.
                # The current 'load_model' signature expects a single 'timeseries' argument.
                # You'll need to decide how to pass 'all_timeseries_data' to it.
                # Example: Pass the dictionary directly if load_model is updated.
                response = load_model(all_timeseries_data, title, indicators, model, tokenizer, processor, None, None) 

                # --- Log the analysis for the entire scenario ---
                # NOTE: You might need to adapt 'analyze_log' as well.
                # It currently expects a single timeseries list and name.
                # Example: Pass the dictionary of lists and the scenario title.
                all_timeseries_lists = {name: ts.tolist() for name, ts in all_timeseries_data.items()}
                analyze_log(log_path, all_timeseries_lists, title, response) # Pass dict and title

            except Exception as e:
                print(f"    Error processing scenario {scenario_info['title']}: {e}")
                # Consider adding more specific error handling if needed

        # --- Release Model Resources (Optional - Uncomment if model was initialized) ---
        print("\nReleasing LLM resources...")
        release_llm()

        print("\nScript finished.")
        
        # for i in range(len(timeseries_data)):
        #     print(timeseries_name)
        #     timeseries = timeseries_data[i]
        #     ts_name = timeseries_name[i]
        #     response = load_model(timeseries)
        #     analyze_log(log_path, timeseries, ts_name, response)