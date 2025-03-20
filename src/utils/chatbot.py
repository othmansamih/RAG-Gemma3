from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from utils.custom_api import CustomAPIEmbeddings, CustomAPILlm
from utils.load_app_config import LoadAppConfig
from typing import List

APP_CONFIG = LoadAppConfig()

class Chatbot:
    @staticmethod
    def user(user_prompt: str, chat_history: List) -> List:
        """
        Adds the user prompt to the chat history.
        
        Args:
            user_prompt (str): The input provided by the user.
            chat_history (List): The existing chat history.
        
        Returns:
            List: Updated chat history including the user's message.
        """
        chat_history.append({"role": "user", "content": user_prompt})
        return chat_history
        
    @staticmethod    
    def bot(user_prompt: str, chat_history: List, type_documents: str) -> tuple[str, str, List]:
        """
        Handles the chatbot's response generation process.
        
        Args:
            user_prompt (str): The input provided by the user.
            chat_history (List): The existing chat history.
            type_documents (str): The namespace defining the type of documents to retrieve.
        
        Returns:
            tuple: A string representation of retrieved documents, an empty string (placeholder), and updated chat history.
        """
        # Initialize vector store with embeddings
        vector_store = PineconeVectorStore(
            embedding=CustomAPIEmbeddings(api_url=APP_CONFIG.embed_api_url),
            index_name=APP_CONFIG.index_name
        )
        
        # Retrieve relevant documents based on user prompt
        retriever = vector_store.as_retriever(search_kwargs={"k": APP_CONFIG.k, "namespace": type_documents})
        retrieved_docs_str = ""
        retrieved_docs = retriever.invoke(user_prompt)
        for i, doc in enumerate(retrieved_docs):
            retrieved_docs_str += f"# Retrieved document {i+1}: \n" + doc.page_content + "\n\n"

        # Create the system and human message prompt templates
        system_message = SystemMessagePromptTemplate.from_template(
            template=APP_CONFIG.system_prompt
        )
        human_message = HumanMessagePromptTemplate.from_template(
            template="User prompt:\n{input}\n\nRetrieved documents:\n{context}"
        )          
        prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # Initialize LLM with API endpoint
        llm = CustomAPILlm(api_url=APP_CONFIG.llm_api_url)
        
        # Create document chain for combining retrieved docs with the prompt
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt_template
        )
        
        # Create retrieval chain for fetching relevant documents and processing response
        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        # Generate response from LLM
        response = rag_chain.invoke({
            "input": user_prompt,
            "context": retrieved_docs_str
        })

        # Append chatbot's response to chat history
        chat_history.append({"role": "assistant", "content": response["answer"]})
        return retrieved_docs_str, "", chat_history
