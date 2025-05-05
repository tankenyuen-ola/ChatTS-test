

def save_log(filepath: str, timeseries_data: dict, title: str, response: str):
    """Append analysis results to a log file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"--- Analysis: {title} ---\n")
        f.write(response + "\n")
        f.write("-"*40 + "\n")