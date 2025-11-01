import os
from dotenv import load_dotenv
from clone import clone_repo
from file_contents import extract_files_to_csv
from issues import extract_issues
import pickle
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from models.gemini_models_rag import (
    load_env_and_configure,
    build_vector_index,
    create_prompt,
    retrieve_relevant_files,
    load_issue,
    start_chat,
)

# =====================================================
# Helper: Let user choose an issue from CSV
# =====================================================
def select_issue_from_csv(csv_path):
    """Display all issues and let user pick one by number."""
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("❌ No issues found in the CSV file.")
        return None

    print("\n🐞 Available Issues:")
    print("=" * 60)

    # Display only title + URL with numbering
    for i, row in df.iterrows():
        print(f"[{i}] {row.get('title', 'Untitled Issue')}")
        if "url" in df.columns:
            print(f"    🔗 {row['url']}")
        print("-" * 60)

    while True:
        try:
            selection = int(input("\n👉 Enter the issue number you want to solve: "))
            if 0 <= selection < len(df):
                print(f"✅ You selected issue #{selection}: {df.loc[selection, 'title']}")
                return selection
            else:
                print(f"⚠️ Please enter a number between 0 and {len(df) - 1}.")
        except ValueError:
            print("⚠️ Please enter a valid number.")


# =====================================================
# Main pipeline
# =====================================================
def run_pipeline(repo_url, clone_dir="cloned_repo"):
    MODEL_NAME = "gemini-2.5-flash-lite"
    FILES_CSV = r"data\repo_files_data.csv"
    ISSUES_CSV = r"data\repo_issues.csv"
    CUSTOM_MODEL_PATH = r"C:\Users\hp\Desktop\prashna\models\all-MiniLM-L6-v2"
    INDEX_PATH = r"embeddings\repo_index.pkl"
    TOP_K = 30

    # Step 1: Clone repo
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(clone_dir, repo_name)
    clone_repo(repo_url, clone_dir)

    # Step 2: Extract files
    print("\n📂 Extracting repository files...")
    extract_files_to_csv(repo_path, FILES_CSV)

    # Step 3: Extract issues
    print("\n🐞 Extracting repository issues...")
    token = os.getenv("GITHUB_TOKEN")
    extract_issues(repo_url, output_file=ISSUES_CSV, token=token)

    print("\n✅ Pipeline completed successfully!")
    print(f"   - Files CSV: {FILES_CSV}")
    print(f"   - Issues CSV: {ISSUES_CSV}")

    # Step 4: Configure environment + embeddings
    load_env_and_configure()

    if not os.path.exists(INDEX_PATH):
        build_vector_index()

    # Step 5: Let user choose issue
    row_index = select_issue_from_csv(ISSUES_CSV)
    if row_index is None:
        print("❌ No valid issue selected. Exiting.")
        return

    issue = load_issue(ISSUES_CSV, row_index=row_index)
    repo_context = retrieve_relevant_files(issue["body"])
    system_prompt = create_prompt(issue, repo_context)

    print(f"\n📂 Loaded Issue #{issue.get('number', 'N/A')} — “{issue.get('title', '')}”")
    start_chat(system_prompt, MODEL_NAME)


# =====================================================
# Entry point
# =====================================================
def main():
    load_dotenv()
    repo_url = input("🔗 Enter GitHub Repository URL: ").strip()

    if not repo_url:
        print("❌ No repository URL provided. Exiting.")
        return

    run_pipeline(repo_url)


if __name__ == "__main__":
    main()
