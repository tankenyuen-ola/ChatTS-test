import os
import numpy as np

# def load_scenarios(base_dir: str):
#     """
#     Load all NPZ and TXT scenarios under base_dir.
#     Returns:
#       dict mapping scenario keys → {metadata…, 'data': np.ndarray},
#       or None if no valid .npz/.txt pairs (or all are empty).
#     """
#     # 1. Gather all files
#     try:
#         entries = os.listdir(base_dir)
#     except FileNotFoundError:
#         return None

#     npz_entries = os.path.join(base_dir, entries[0])
#     txt_entries = os.path.join(base_dir, entries[1])

#     all_npz  = os.listdir(npz_entries)         
#     all_txt  = os.listdir(txt_entries) 
#     # If there are no NPZ or TXT files at all, bail out:
#     npz_files = [f for f in all_npz if f.lower().endswith('.npz')]
#     txt_files = {os.path.splitext(f)[0] for f in all_txt if f.lower().endswith('.txt')}
#     if not npz_files and not txt_files:
#         return None

#     scenarios = {}
#     for fname in npz_files:
#         key = os.path.splitext(fname)[0]
#         npz_path = os.path.join(npz_entries, fname)
#         print(npz_path)

#         # 2. Load the array; skip if empty
#         try:
#             data = np.load(npz_path)
#         except Exception:
#             # corrupted or unreadable .npz
#             continue

#         # in case the .npz archive is empty or array has no elements
#         if getattr(data, 'size', 0) == 0:
#             print('failed')
#             continue

#         # 3. Read associated .txt metadata if present
#         metadata = {}
#         txt_path = os.path.join(txt_files, key + '.txt')
#         print(txt_path)
#         if os.path.exists(txt_path):
#             with open(txt_path, 'r', encoding='utf-8') as tf:
#                 for line in tf:
#                     if ':' in line:
#                         k, v = line.strip().split(':', 1)
#                         metadata[k.strip()] = v.strip()

#         scenarios[key] = {**metadata, 'data': data}

#     # 4. If after all that we still have no scenarios, return None
#     return scenarios or None

def load_scenarios(base_dir: str):
    """
    Loads time series data and metadata from all matching npz/txt pairs in a directory.

    Args:
        base_dir (str): Path to the directory containing .npz and .txt files.

    Returns:
        dict: A dictionary where keys are scenario base names (e.g., 'cicd')
              and values are dictionaries containing 'title', 'metadata_description',
              and 'data' (the time series arrays from the npz file).
              Returns an empty dictionary if the directory is not found or no matching pairs exist.
    """
    all_scenario_data = {}

    if not os.path.isdir(base_dir):
        print(f"Error: Data directory not found at {base_dir}")
        return all_scenario_data

    # Find all npz files in the directory
    npz_dir = os.path.join(base_dir, 'data')
    meta_dir = os.path.join(base_dir, 'meta')
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
    
    return all_scenario_data