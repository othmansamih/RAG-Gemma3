from utils.upload_document import UploadDocuemnt
from utils.chatbot import Chatbot
from utils.clean_chatbot import CleanChatbot
from utils.ui_settings import UISettings
import gradio as gr

# Clean up previous chatbot data and uploaded documents
CleanChatbot.remove_uploaded_documents_directory()
CleanChatbot.remove_uploaded_documents_namespace()

# Define the Gradio UI
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("RAG-DeepSeek"):
            with gr.Row():
                with gr.Column(visible=False) as references_bar:
                    references = gr.Markdown(
                        value="References",
                        height=330
                    )  
                
                with gr.Column():
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=330,
                        group_consecutive_messages=False
                    )
            
            with gr.Row():
                text_input = gr.TextArea(
                    lines=4,
                    placeholder="Enter your prompt here",
                    container=False
                )
            
            with gr.Row():
                submit_btn = gr.Button(
                    value="Submit"
                )
                sidebar_state = gr.State(False)
                references_btn = gr.Button(
                    value="References"
                )

                rag_with_dropdown = gr.Dropdown(
                    choices=[
                        "Pre-processed documents",
                        "Uploaded document(s)"
                    ],
                    label="RAG with",
                    value="Pre-processed documents",
                    interactive=True
                )
                upload_btn = gr.UploadButton(
                    label="Upload your doc(s)",
                    file_count="multiple",
                    file_types=[".pdf"]
                )
                clear_btn = gr.ClearButton(
                    components=[references, chatbot, text_input]
                )
            
            # Define chatbot interactions
            submit_btn.click(fn=Chatbot.user,
                             inputs=[text_input, chatbot],
                             outputs=chatbot,
                             queue=False).then(fn=Chatbot.bot,
                                               inputs=[text_input, chatbot, rag_with_dropdown],
                                               outputs=[references, text_input, chatbot],
                                               queue=False)
            
            text_input.submit(fn=Chatbot.user,
                              inputs=[text_input, chatbot],
                              outputs=chatbot,
                              queue=False).then(fn=Chatbot.bot,
                                                inputs=[text_input, chatbot, rag_with_dropdown],
                                                outputs=[references, text_input, chatbot],
                                                queue=False)
            
            # Toggle references sidebar
            references_btn.click(fn=UISettings.toggle_sidebar,
                                 inputs=sidebar_state,
                                 outputs=[references_bar, sidebar_state])
            
            # Handle document upload
            upload_btn.upload(fn=UploadDocuemnt.process_uploaded_documents, 
                              inputs=[upload_btn, rag_with_dropdown, chatbot],
                              outputs=[text_input, chatbot],
                              queue=False)
            
# Launch the Gradio UI
if __name__ == "__main__":
    demo.launch()
