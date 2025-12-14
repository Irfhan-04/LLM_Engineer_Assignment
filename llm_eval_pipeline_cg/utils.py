import json

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_latest_turn(conversation: dict):
    messages = conversation.get("messages", [])
    user, assistant = "", ""

    for msg in reversed(messages):
        if msg.get("role") == "assistant" and not assistant:
            assistant = msg.get("content", "")
        elif msg.get("role") == "user" and not user:
            user = msg.get("content", "")
        if user and assistant:
            break

    return user, assistant
