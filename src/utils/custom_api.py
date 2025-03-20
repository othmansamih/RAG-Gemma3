from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import Field
from utils.utilities import convert_messages_to_dict, count_tokens
from utils.load_app_config import LoadAppConfig
import requests
from typing import Any, List, Optional
import json


APP_CONFIG = LoadAppConfig()

class CustomAPIEmbeddings(Embeddings):
    """
    A custom embedding model that interacts with an external API to generate embeddings.
    """
    def __init__(self, api_url: str) -> None:
        """
        Initializes the CustomAPIEmbeddings class.

        Args:
            api_url (str): The URL of the embedding API.
        """
        self.api_url = api_url
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single query text.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: The embedding vector.
        """
        data = {"text": text}
        response = requests.post(self.api_url, json=data)
        return response.json()["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for multiple documents.

        Args:
            texts (List[str]): A list of texts to be embedded.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        embeddings = [self.embed_query(text) for text in texts]
        return embeddings
    

class CustomAPILlm(BaseChatModel):
    """
    A custom language model that interacts with an external API for generating chat responses.
    """
    api_url: str = Field(..., description="API URL for the LLM")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generates a chat completion using the external API.

        Args:
            messages (List[BaseMessage]): A list of messages in the conversation.
            stop (Optional[List[str]], optional): List of stop words. Defaults to None.
            run_manager (Optional[Any], optional): Internal runtime manager. Defaults to None.

        Returns:
            ChatResult: The generated chat response.
        """
        # Convert messages to API-compatible format
        formatted_messages = convert_messages_to_dict(messages)
        data = {"messages": formatted_messages}
            
        # Send request to API
        response = requests.post(url=self.api_url, json=data)
        response_text = response.json()["response"]

        # Calculate token usage
        ct_input_tokens = count_tokens(messages)
        ct_output_tokens = len(response_text)

        # Create AIMessage object with metadata
        message = AIMessage(
            content=response_text,
            additional_kwargs={},
            response_metadata={"time_in_seconds": 3},
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )
        
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        """
        Returns the type of language model being used.
        
        Returns:
            str: The model ID from the configuration.
        """
        return f"{APP_CONFIG.gen_model_id}"