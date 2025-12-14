"""
JSON Parser Utility

Handles parsing of conversation and context JSON files.
Supports multiple formats to be flexible with input structures.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger("JSONParser")


def parse_conversation(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse conversation JSON into standardized format.
    
    Supports multiple formats:
    1. Standard format: {"messages": [...]}
    2. Chat format: {"conversation": [...]}
    3. Direct array: [...]
    
    Args:
        conversation_data: Raw conversation JSON
        
    Returns:
        Standardized conversation dictionary
    """
    try:
        # Format 1: Standard messages format
        if "messages" in conversation_data:
            return {
                "messages": normalize_messages(conversation_data["messages"]),
                "metadata": conversation_data.get("metadata", {})
            }
        
        # Format 2: Conversation key
        if "conversation" in conversation_data:
            return {
                "messages": normalize_messages(conversation_data["conversation"]),
                "metadata": conversation_data.get("metadata", {})
            }
        
        # Format 3: Direct array
        if isinstance(conversation_data, list):
            return {
                "messages": normalize_messages(conversation_data),
                "metadata": {}
            }
        
        # Format 4: Single conversation object
        if "id" in conversation_data and "content" in conversation_data:
            return {
                "messages": normalize_messages([conversation_data]),
                "metadata": {}
            }
        
        logger.warning("Unrecognized conversation format, using as-is")
        return {"messages": [], "metadata": {}}
        
    except Exception as e:
        logger.error(f"Failed to parse conversation: {str(e)}")
        raise ValueError(f"Invalid conversation format: {str(e)}")


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize message format to standard structure.
    
    Standard format:
    {
        "id": str,
        "role": "user" | "assistant" | "system",
        "content": str,
        "timestamp": str (optional)
    }
    """
    normalized = []
    
    for i, msg in enumerate(messages):
        # Handle different role naming conventions
        role = msg.get("role", msg.get("sender", msg.get("type", "unknown")))
        role = normalize_role(role)
        
        # Handle different content field names
        content = msg.get("content", msg.get("message", msg.get("text", "")))
        
        # Generate ID if not present
        msg_id = msg.get("id", msg.get("message_id", f"msg_{i}"))
        
        normalized.append({
            "id": msg_id,
            "role": role,
            "content": str(content),
            "timestamp": msg.get("timestamp", msg.get("created_at", ""))
        })
    
    return normalized


def normalize_role(role: str) -> str:
    """Normalize role names to standard values."""
    role_lower = role.lower()
    
    # Map various role names to standard ones
    if role_lower in ["user", "human", "customer"]:
        return "user"
    elif role_lower in ["assistant", "ai", "bot", "agent", "system_response"]:
        return "assistant"
    elif role_lower in ["system"]:
        return "system"
    else:
        return "assistant"  # Default to assistant


def parse_context(context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse context vectors JSON into standardized format.
    
    Supports multiple formats:
    1. {"contexts": [...]}
    2. {"retrieved_documents": [...]}
    3. {"chunks": [...]}
    4. Direct array: [...]
    
    Args:
        context_data: Raw context JSON
        
    Returns:
        List of context dictionaries
    """
    try:
        # Format 1: contexts key
        if "contexts" in context_data:
            return normalize_contexts(context_data["contexts"])
        
        # Format 2: retrieved_documents
        if "retrieved_documents" in context_data:
            return normalize_contexts(context_data["retrieved_documents"])
        
        # Format 3: chunks
        if "chunks" in context_data:
            return normalize_contexts(context_data["chunks"])
        
        # Format 4: documents
        if "documents" in context_data:
            return normalize_contexts(context_data["documents"])
        
        # Format 5: results
        if "results" in context_data:
            return normalize_contexts(context_data["results"])
        
        # Format 6: Direct array
        if isinstance(context_data, list):
            return normalize_contexts(context_data)
        
        logger.warning("Unrecognized context format, trying as single context")
        return normalize_contexts([context_data])
        
    except Exception as e:
        logger.error(f"Failed to parse context: {str(e)}")
        raise ValueError(f"Invalid context format: {str(e)}")


def normalize_contexts(contexts: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize context items to standard structure.
    
    Standard format:
    {
        "id": str,
        "text": str,
        "score": float (optional),
        "metadata": dict (optional)
    }
    """
    normalized = []
    
    for i, ctx in enumerate(contexts):
        # Handle string contexts
        if isinstance(ctx, str):
            normalized.append({
                "id": f"ctx_{i}",
                "text": ctx,
                "content": ctx,  # Alias for compatibility
                "score": None,
                "metadata": {}
            })
            continue
        
        # Handle dict contexts
        if isinstance(ctx, dict):
            # Extract text content (try multiple field names)
            text = (
                ctx.get("text") or 
                ctx.get("content") or 
                ctx.get("page_content") or 
                ctx.get("chunk") or
                ctx.get("document") or
                str(ctx)
            )
            
            # Extract score/relevance
            score = (
                ctx.get("score") or 
                ctx.get("similarity") or 
                ctx.get("relevance") or
                ctx.get("distance")
            )
            
            normalized.append({
                "id": ctx.get("id", f"ctx_{i}"),
                "text": str(text),
                "content": str(text),  # Alias
                "score": float(score) if score is not None else None,
                "metadata": ctx.get("metadata", {})
            })
    
    return normalized


def validate_inputs(
    conversation_data: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> bool:
    """
    Validate that inputs are properly formatted.
    
    Args:
        conversation_data: Parsed conversation data
        context_data: Parsed context data
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check conversation has messages
    if not conversation_data.get("messages"):
        raise ValueError("Conversation must contain at least one message")
    
    # Check for at least one AI response
    has_ai_response = any(
        msg.get("role") == "assistant" 
        for msg in conversation_data["messages"]
    )
    if not has_ai_response:
        raise ValueError("Conversation must contain at least one AI response")
    
    # Check context is not empty
    if not context_data:
        logger.warning("Context data is empty - hallucination check will be limited")
    
    return True