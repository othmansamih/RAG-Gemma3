from flask import Flask, request, jsonify, Response
from transformers import pipeline, AutoModel, Gemma3ForCausalLM, AutoTokenizer
from utils.load_app_config import LoadAppConfig
import torch

# Load application configuration
APP_CONFIG = LoadAppConfig()
app = Flask(__name__)

# Text generation model initialization
gen_model_id = APP_CONFIG.gen_model_id
gen_model = Gemma3ForCausalLM.from_pretrained(gen_model_id).eval()
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
generator = pipeline(
    task="text-generation", 
    model=gen_model, 
    tokenizer=gen_tokenizer,
    device_map=APP_CONFIG.device_map
)


# Embedding model initialization
embed_model_id = APP_CONFIG.embed_model_id
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
embed_model = AutoModel.from_pretrained(embed_model_id)

@app.route("/generate", methods=["POST"])
def generate_text() -> Response:
    """
    Generate text based on input messages using a pre-trained language model.
    
    Request:
        - JSON object with a "messages" field containing the input text.
    
    Response:
        - JSON object with the generated text under the "response" field.
    """
    data = request.json
    messages = data.get("messages", "")
    response = generator(
        messages,
        temperature=APP_CONFIG.temperature,
        do_sample=True,
        max_new_tokens=APP_CONFIG.max_new_tokens
    )
    return jsonify({"response": response[0]["generated_text"][-1]["content"]})

@app.route("/embed", methods=["POST"])
def generate_embedding() -> Response:
    """
    Generate text embeddings for a given input text using a pre-trained model.
    
    Request:
        - JSON object with a "text" field containing the input text.
    
    Response:
        - JSON object with the computed embedding as a list of float values under the "embedding" field.
    """
    # get the input text
    data = request.json
    text = data.get("text", "")
    
    # Tokenize input text
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    return jsonify({"embedding": embeddings.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
