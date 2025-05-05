import json
import os

HISTORY_PATH = os.path.join("chat_history", "chat_history.json")


def save_chat_history(chat_history, path: str = HISTORY_PATH):
    """Persist chat history to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)


def update_chat_history(chat_history: list, user_msg: str, model_resp: str):
    """Append user message and model response to history list."""
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": model_resp})
    return chat_history