from pinecone.grpc import PineconeGRPC as Pinecone
from utils.load_app_config import LoadAppConfig
import shutil
import os

# Load application configuration
APP_CONFIG = LoadAppConfig()

class CleanChatbot:
    """
    A utility class for cleaning chatbot-related resources, such as
    uploaded documents and their associated vector database namespaces.
    """
    
    @staticmethod
    def remove_uploaded_documents_namespace() -> None:
        """
        Removes the namespace associated with uploaded documents from the Pinecone vector database.
        This effectively deletes all stored embeddings related to uploaded documents.
        """
        pc = Pinecone()
        index = pc.Index(APP_CONFIG.index_name)
        stats = index.describe_index_stats()
        
        if "namespaces" in stats and "Uploaded document(s)" in stats["namespaces"]:
            index.delete(delete_all=True, namespace="Uploaded document(s)")
    
    @staticmethod
    def remove_uploaded_documents_directory() -> None:
        """
        Deletes the directory where uploaded documents are stored.
        This removes all uploaded files from the local storage.
        """
        if os.path.exists(APP_CONFIG.uploaded_documents_dir):
            shutil.rmtree(APP_CONFIG.uploaded_documents_dir)