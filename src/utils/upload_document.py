from utils.prepare_vectordb import PrepareVectorDB
from utils.load_app_config import LoadAppConfig
from typing import List, Tuple
import shutil
import os

# Load application configuration
APP_CONFIG = LoadAppConfig()

class UploadDocuemnt():
    """
    A class for handling document uploads and processing them into a vector database.
    """
    
    @staticmethod
    def process_uploaded_documents(uploaded_documents: List[str], type_documents: str, chat_history: List[dict]) -> Tuple[str, List[dict]]:
        """
        Process and store uploaded documents in a vector database.
        
        Args:
            uploaded_documents (List[str]): List of file paths for uploaded documents.
            type_documents (str): Specifies the type of documents (should be "Uploaded document(s)").
            chat_history (List[dict]): List representing the chat history.
        
        Returns:
            Tuple[str, List[dict]]: An empty string and the updated chat history.
        """
        if type_documents == "Uploaded document(s)":
            os.makedirs(APP_CONFIG.uploaded_documents_dir, exist_ok=True)
            for uploaded_docuemnt in uploaded_documents:
                shutil.move(uploaded_docuemnt, APP_CONFIG.uploaded_documents_dir)

            # Prepare and store documents in the vector database
            prepare_vectordb = PrepareVectorDB(
                APP_CONFIG.uploaded_documents_dir,
                APP_CONFIG.chunk_size,
                APP_CONFIG.chunk_overlap,
                APP_CONFIG.embed_model_id,
                APP_CONFIG.index_name,
                APP_CONFIG.cloud,
                APP_CONFIG.region
            )
            prepare_vectordb.prepare_and_save_vectordb(namespace="Uploaded document(s)")
            
            chat_history.append({"role": "assistant", "content": "The document(s) have been successfully uploaded!"})
        else:
            chat_history.append({"role": "assistant", "content": "You should first click on 'Upload Document(s)' in the rag dropdown to upload document(s)."})
        
        return "", chat_history
