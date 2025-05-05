from prompt.template import *
import numpy as np
import torch
from vllm import SamplingParams

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
        prompt = batch_prompt.format(num_valid_series=num_valid_series, title=title, valid_indicator_string=valid_indicator_string, ts_placeholders=ts_placeholders)
    else:
        prompt = interactive_prompt.format(num_valid_series=num_valid_series, title=title, valid_indicator_string=valid_indicator_string, ts_placeholders=ts_placeholders, chat_history=chat_history[:-1], question=question)
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
            #print(outputs)
            response = outputs[0].outputs[0].text
            response = response.removeprefix('[]\n\n')
            return response
    except Exception as e:
        print(f"Error during model generation: {e}")
        return f"Error generating response: {e}"