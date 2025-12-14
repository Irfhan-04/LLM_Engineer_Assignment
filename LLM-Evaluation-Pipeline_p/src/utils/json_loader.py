"""JSON loader and validator."""

import json
import logging
from typing import Dict, Any, Tuple
from pathlib import Path


logger = logging.getLogger(__name__)


class JSONValidator:
    """Validates JSON structure."""

    @staticmethod
    def validate_conversation(conversation: Dict[str, Any]) -> bool:
        """Validate conversation JSON."""
        required = ["conversation_id", "messages"]

        if not all(field in conversation for field in required):
            raise ValueError(f"Missing required fields: {required}")

        if not isinstance(conversation["messages"], list):
            raise ValueError("Messages must be a list")

        for msg in conversation["messages"]:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message needs 'role' and 'content'")

        return True

    @staticmethod
    def validate_context_vectors(context: Dict[str, Any]) -> bool:
        """Validate context JSON."""
        required = ["message_id", "vectors"]

        if not all(field in context for field in required):
            raise ValueError(f"Missing required fields: {required}")

        if not isinstance(context["vectors"], list):
            raise ValueError("Vectors must be a list")

        return True


class JSONLoader:
    """Loads and processes JSON files."""

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON from file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @staticmethod
    def load_conversation_and_context(
        conv_path: str, context_path: str
    ) -> Tuple[Dict, Dict]:
        """Load and validate both JSONs."""
        conversation = JSONLoader.load_json(conv_path)
        context = JSONLoader.load_json(context_path)

        JSONValidator.validate_conversation(conversation)
        JSONValidator.validate_context_vectors(context)

        return conversation, context

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """Save data to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
