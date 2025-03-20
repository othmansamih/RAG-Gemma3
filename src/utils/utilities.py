from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from utils.load_app_config import LoadAppConfig
from typing import List

# Load application configuration
APP_CONFIG = LoadAppConfig()

def convert_messages_to_dict(messages: List[BaseMessage]) -> List[dict]:
    """
    Convert a list of LangChain messages into a dictionary format.
    
    Args:
        messages (List[BaseMessage]): A list of messages, which can be of type SystemMessage, HumanMessage, or other assistant messages.
    
    Returns:
        List[dict]: A list of dictionaries, each containing a "role" (system, user, or assistant) and "content" (message text).
    """
    formatted_messages = []
    for message in messages:
        role = "system" if isinstance(message, SystemMessage) else "user" if isinstance(message, HumanMessage) else "assistant"
        formatted_messages.append({"role": role, "content": message.content})
    return formatted_messages


def count_tokens(messages: List[BaseMessage]) -> int:
    """
    Count the total number of tokens in a list of messages using a specified tokenizer.
    
    Args:
        messages (List[BaseMessage]): A list of messages.

    Returns:
        int: The total number of tokens across all messages.
    """
    total_tokens = 0
    for message in messages:
        total_tokens += len(message.content)
    return total_tokens
