from utils.prepare_vectordb import PrepareVectorDB
from utils.load_app_config import LoadAppConfig

# Load application configuration
APP_CONFIG = LoadAppConfig()

def process_documents_manually() -> None:
    """
    Process and store documents in a vector database.
    
    This function initializes the vector database preparation process using configuration parameters.
    It reads documents from the specified directory, chunks them according to the defined size and overlap,
    embeds them using a specified model, and saves them to the vector database.
    
    Namespace: "Pre-processed documents"
    """
    prepare_vectordb = PrepareVectorDB(
        APP_CONFIG.documents_dir,
        APP_CONFIG.chunk_size,
        APP_CONFIG.chunk_overlap,
        APP_CONFIG.embed_model_id,
        APP_CONFIG.index_name,
        APP_CONFIG.cloud,
        APP_CONFIG.region
    )
    prepare_vectordb.prepare_and_save_vectordb(namespace="Pre-processed documents")

if __name__ == "__main__":
    process_documents_manually()