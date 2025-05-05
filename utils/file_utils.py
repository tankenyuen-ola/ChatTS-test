import os

def file_duplicate(filepath: str) -> str:
    """If filepath exists, append a counter suffix before extension."""
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_path = filepath
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path