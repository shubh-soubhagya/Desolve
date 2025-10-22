import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

# API key for Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🧠 MODEL REGISTRY
AVAILABLE_MODELS = {
    # ---- GROQ MODELS ----
    "groq:llama3-70b": {
        "type": "groq",
        "description": "LLaMA 3 (70B) — high-performance open model on Groq."
    },
    "groq:llama3-8b": {
        "type": "groq",
        "description": "LLaMA 3 (8B) — efficient, smaller variant hosted on Groq."
    },
    "groq:gemma2-9b": {
        "type": "groq",
        "description": "Gemma 2 (9B) — strong reasoning and summarization model."
    },
    "groq:mixtral-8x7b": {
        "type": "groq",
        "description": "Mixtral 8x7B — mixture-of-experts model for structured code understanding."
    },

    # ---- HUGGING FACE MODELS ----
    "hf:bart-summarizer": {
        "type": "huggingface",
        "description": "BART-based summarizer for issue and code documentation."
    },
    "hf:codebert": {
        "type": "huggingface",
        "description": "CodeBERT — open-source model for code understanding and embeddings."
    },
    "hf:distilbart-cnn": {
        "type": "huggingface",
        "description": "DistilBART — fast summarizer for lightweight text analysis."
    },

    # ---- LOCAL / CUSTOM MODELS ----
    "local:ollama-llama3": {
        "type": "local",
        "description": "Run LLaMA 3 locally using Ollama (no API key required)."
    },
    "local:custom-finetuned": {
        "type": "local",
        "description": "Your fine-tuned model for repository-specific analysis."
    },
}


# 📋 LIST AVAILABLE MODELS
def list_available_models():
    """Display all available models with descriptions."""
    print("\n🧠 Available Models:")
    for name, info in AVAILABLE_MODELS.items():
        print(f" - {name:<25} → {info['description']}")


# ⚙️ MODEL LOADING
def load_model(model_name: str):
    """
    Load and initialize the selected model.
    Supports Groq API, Hugging Face models, and local models.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"❌ Model '{model_name}' not found. Use list_available_models() to view options.")

    model_type = AVAILABLE_MODELS[model_name]["type"]

    # -------------------- GROQ MODELS --------------------
    if model_type == "groq":
        def groq_infer(prompt: str):
            if not GROQ_API_KEY:
                raise ValueError("❌ Missing GROQ_API_KEY in environment variables.")
            model_id = model_name.split(":")[1]
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 1000
                },
            )
            if response.status_code != 200:
                raise RuntimeError(f"Groq API Error: {response.text}")
            return response.json()["choices"][0]["message"]["content"]
        return groq_infer

    # -------------------- HUGGING FACE MODELS --------------------
    elif model_type == "huggingface":
        model_key = model_name.split(":")[1]

        if "bart" in model_key:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            return lambda text: summarizer(text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]

        elif "distilbart" in model_key:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            return lambda text: summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]

        elif "codebert" in model_key:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base")
            return (tokenizer, model)

    # -------------------- LOCAL / CUSTOM MODELS --------------------
    elif model_type == "local":
        def local_infer(prompt: str):
            print(f"⚙️ Using local model: {model_name}")
            try:
                # Example: Ollama CLI call or custom local API
                response = os.popen(f"ollama run {model_name.split(':')[1]} '{prompt}'").read()
                return response.strip() or "⚠️ No response from local model."
            except Exception as e:
                return f"⚠️ Local model error: {str(e)}"
        return local_infer

    else:
        raise ValueError("⚠️ Unsupported model type.")


# 🧩 INTERACTIVE MODEL SELECTION
def choose_model_interactively():
    """Let the user pick a model from the available list."""
    list_available_models()
    choice = input("\n💬 Enter the model name you want to use: ").strip()
    try:
        model = load_model(choice)
        print(f"✅ Model '{choice}' loaded successfully.")
        return model, choice
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


# 🧠 TEST MODE
if __name__ == "__main__":
    model, model_name = choose_model_interactively()
    if model:
        test_prompt = "Summarize what this GitHub repository does."
        print("\n🧾 Model Output:\n")
        print(model(test_prompt))
