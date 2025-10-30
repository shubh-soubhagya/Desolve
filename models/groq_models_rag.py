import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# =====================================================
# CONFIGURATION
# =====================================================
# MODEL_NAME = "openai/gpt-oss-120b"   # You can use "gemma-7b-it" or "meta-llama/llama-guard-4-12b"
# FILES_CSV = "repo_files_data.csv"
# ISSUES_CSV = "repo_issues.csv"
# INDEX_PATH = "repo_index.pkl"
# CUSTOM_MODEL_PATH = "all-MiniLM-L6-v2"   # 🔹 change to your embedding model path
# TOP_K = 15  # Number of most relevant files to retrieve per query


MODEL_NAME = "openai/gpt-oss-120b"   # You can use "gemma-7b-it" or "meta-llama/llama-guard-4-12b"
FILES_CSV = r"C:\Users\hp\Desktop\MinorProj\Desolve\repo_files_data.csv"
ISSUES_CSV = r"C:\Users\hp\Desktop\MinorProj\Desolve\repo_issues.csv"
INDEX_PATH = "repo_index.pkl"
CUSTOM_MODEL_PATH = r"C:\Users\hp\Desktop\prashna\models\all-MiniLM-L6-v2"   # 🔹 change to your embedding model path
TOP_K = 15  # Number of most relevant files to retrieve per query

# =====================================================
# ENVIRONMENT SETUP
# =====================================================
def load_env_and_configure():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env file.")
        exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    return client


# =====================================================
# BUILD VECTOR INDEX (RUN ONCE)
# =====================================================
def build_vector_index():
    print("🧠 Building vector index from repo files...")

    df = pd.read_csv(FILES_CSV, encoding="utf-8")
    if df.empty:
        print("❌ No files found in repo_files_data.csv")
        exit(1)

    print(f"📄 Total files: {len(df)}")

    model = SentenceTransformer(CUSTOM_MODEL_PATH)  # 🔹 custom model path
    embeddings = model.encode(df["file_content"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump((index, df), f)

    print(f"✅ Vector index saved at: {INDEX_PATH}")


# =====================================================
# LOAD INDEX & RETRIEVE FILES
# =====================================================
def retrieve_relevant_files(query: str, top_k: int = TOP_K):
    if not os.path.exists(INDEX_PATH):
        print("⚠️ Index not found. Building one now...")
        build_vector_index()

    with open(INDEX_PATH, "rb") as f:
        index, df = pickle.load(f)

    model = SentenceTransformer(CUSTOM_MODEL_PATH)
    query_vec = model.encode([query], convert_to_numpy=True)

    D, I = index.search(query_vec, top_k)
    top_files = df.iloc[I[0]].to_dict(orient="records")

    repo_context = ""
    for file in top_files:
        repo_context += (
            f"\n\n### File: {file.get('file_name', '')} ({file.get('file_path', '')})\n"
            f"```{file.get('file_extension', '')}\n"
            f"{file.get('file_content', '')[:3000]}\n```\n"
        )
    return repo_context


# =====================================================
# CREATE PROMPT
# =====================================================
def create_prompt(issue: dict, repo_context: str) -> str:
    issue_title = issue.get("title", "Untitled Issue")
    issue_body = issue.get("body", "No description provided.")
    issue_number = issue.get("number", "N/A")

    return f"""
You are Desolve AI — an advanced AI code assistant that helps developers fix issues in repositories.

### Issue #{issue_number}: {issue_title}
{issue_body}

### Relevant Repository Files
{repo_context}

Task:
- Analyze and propose precise code fixes.
- Base reasoning only on provided file context.
- Show updated code snippets when possible.
- Maintain professional formatting.
"""


# =====================================================
# LOAD ISSUE
# =====================================================
def load_issue(issue_csv: str, row_index: int = 0):
    df = pd.read_csv(issue_csv)
    if len(df) <= row_index:
        print(f"❌ CSV has only {len(df)} issues. Row {row_index + 1} not found.")
        exit(1)
    return df.to_dict(orient="records")[row_index]


# =====================================================
# CHAT INTERFACE
# =====================================================
def start_chat(client, model_name, system_prompt):
    print("\n🤖 Desolve AI: Ready to discuss and fix your issue!\n")

    history = [
        {"role": "system", "content": "You are Desolve AI — a professional AI developer assistant."},
        {"role": "user", "content": system_prompt},
    ]

    while True:
        user_input = input("👨‍💻 You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting chat. Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=history,
                temperature=0.6,
                max_tokens=2048,
                top_p=1,
                stream=False,
            )

            answer = response.choices[0].message.content
            print("\n🤖 Desolve AI:\n")
            print(answer)
            print("\n" + "-" * 100 + "\n")

            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("💡 Tip: If this happens often, reduce retrieved file count or chunk file content.")
            continue


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    client = load_env_and_configure()

    # Build index (only first time)
    if not os.path.exists(INDEX_PATH):
        build_vector_index()

    issue = load_issue(ISSUES_CSV, row_index=3)

    # Retrieve only relevant files
    repo_context = retrieve_relevant_files(issue["body"])
    system_prompt = create_prompt(issue, repo_context)

    print(f"\n📂 Loaded Issue #{issue.get('number', 'N/A')} — “{issue.get('title', '')}”")
    start_chat(client, MODEL_NAME, system_prompt)
