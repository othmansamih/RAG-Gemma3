from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List
import time
import os

class PrepareVectorDB:
    """
    A class for preparing and storing documents in a Pinecone vector database.
    """
    
    def __init__(self,
                 documents_dir: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 embeddings_model_name: str,
                 index_name: str,
                 cloud: str,
                 region: str
        ) -> None:
        """
        Initializes the PrepareVectorDB class.
        
        Args:
            documents_dir (str): Directory containing the documents to process.
            chunk_size (int): Size of text chunks for splitting documents.
            chunk_overlap (int): Overlap size between chunks.
            embeddings_model_name (str): Name of the HuggingFace embeddings model.
            index_name (str): Name of the Pinecone index.
            cloud (str): Cloud provider for Pinecone.
            region (str): Region of the Pinecone server.
        """
        self.documents_dir = documents_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_name
        )
        self.index_name = index_name
        self.__create_index(index_name, cloud, region)

    def __create_index(self, index_name: str, cloud: str, region: str) -> Index:
        """
        Creates a Pinecone index if it does not already exist.
        
        Args:
            index_name (str): Name of the Pinecone index.
            cloud (str): Cloud provider.
            region (str): Server region.
        
        Returns:
            Index: The created or retrieved Pinecone index.
        """
        print("1- Creating the vectordb index...")
        pc = Pinecone()
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        index = pc.Index(index_name)
        return index

    def __get_all_docs(self) -> List[Document]:
        """
        Loads all documents from the specified directory.
        
        Returns:
            List[Document]: A list of loaded documents.
        """
        print("2- Loading all documents...")
        all_docs = []
        for document in os.listdir(self.documents_dir):
            print(f"\t- The document '{document}' has been loaded successfully!")
            document_path = os.path.join(self.documents_dir, document)
            loader = PyPDFLoader(document_path)
            all_docs.extend(loader.load())
        return all_docs

    def __get_all_splits(self, all_docs: List[Document]) -> List[Document]:
        """
        Splits all loaded documents into smaller text chunks.
        
        Args:
            all_docs (List[Document]): List of loaded documents.
        
        Returns:
            List[Document]: A list of text chunks.
        """
        print("3- Splitting all documents into chunks...")
        all_splits = self.text_splitter.split_documents(all_docs)
        return all_splits

    def prepare_and_save_vectordb(self, namespace: str) -> PineconeVectorStore:
        """
        Prepares and saves document embeddings into the Pinecone vector database.
        
        Args:
            namespace (str): Namespace in the vector database for storing embeddings.
        
        Returns:
            PineconeVectorStore: The created vector store.
        """
        all_docs = self.__get_all_docs()
        all_splits = self.__get_all_splits(all_docs)
        print("4- Saving the chunks in the vectordb...\n")
        vector_store = PineconeVectorStore.from_documents(
            all_splits,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=namespace
        )
        return vector_store
