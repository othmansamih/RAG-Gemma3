from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class LoadAppConfig:
    """
    A class to load application configuration from a YAML file.
    """
    
    def __init__(self) -> None:
        """
        Initializes the LoadAppConfig class by loading the configuration file.
        """
        with open("configs/app_config.yaml") as file:
            app_config = yaml.safe_load(file)
        
        # Directories
        self.documents_dir = app_config["directories"]["documents_dir"]
        self.uploaded_documents_dir = app_config["directories"]["uploaded_documents_dir"]

        # VectorDB Configuration
        self.chunk_size = app_config["vectordb_config"]["text_splitter"]["chunk_size"]
        self.chunk_overlap = app_config["vectordb_config"]["text_splitter"]["chunk_overlap"]
        self.index_name = app_config["vectordb_config"]["index"]["index_name"]
        self.cloud = app_config["vectordb_config"]["index"]["cloud"]
        self.region = app_config["vectordb_config"]["index"]["region"]
        self.k = app_config["vectordb_config"]["retrieved_docs"]["k"]

        # Embedding Model Configuration
        self.embed_model_id = app_config["embeddings"]["embed_model_id"]

        # LLM Configuration
        self.gen_model_id = app_config["llm"]["gen_model_id"]
        self.temperature = app_config["llm"]["temperature"]
        self.device_map = app_config["llm"]["device_map"]
        self.max_new_tokens = app_config["llm"]["max_new_tokens"]
        self.system_prompt = app_config["llm"]["system_prompt"]

        # API URLs
        self.llm_api_url = app_config["api_url"]["llm_api_url"]
        self.embed_api_url = app_config["api_url"]["embed_api_url"]