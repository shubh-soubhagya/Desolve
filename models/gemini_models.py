import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# =====================================================
# Configuration variables (can be edited externally)
# =====================================================
MODEL_NAME = "gemini-2.5-flash-lite"
ISSUES_CSV = r"C:\Users\hp\Desktop\MinorProj\Desolve\repo_issues.csv"
FILES_CSV = r"C:\Users\hp\Desktop\MinorProj\Desolve\repo_files_data.csv"
ROW_INDEX = 3  # 4th issue (0-based index)

# =====================================================
# Load environment variables
# =====================================================
def load_env_and_configure():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GOOGLE_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env file.")
        exit(1)

    genai.configure(api_key=GOOGLE_API_KEY)
    return GOOGLE_API_KEY


# =====================================================
# Load CSVs safely using pandas
# =====================================================
def load_repo_data(issues_csv: str, files_csv: str):
    """Loads repo issues and file data as lists of dicts."""
    issues_df = pd.read_csv(issues_csv, encoding="utf-8")
    files_df = pd.read_csv(files_csv, encoding="utf-8")

    if issues_df.empty:
        print("❌ No issues found in repo_issues.csv.")
        exit(1)

    return issues_df.to_dict(orient="records"), files_df.to_dict(orient="records")


# =====================================================
# Build repository context for Gemini
# =====================================================
def build_repo_context(files: list) -> str:
    """Combines all file data into a context string for Gemini."""
    repo_context = ""
    for file in files:
        repo_context += (
            f"\n\n### File: {file.get('file_name', '')} ({file.get('file_path', '')})\n"
            f"```{file.get('file_extension', '')}\n"
            f"{file.get('file_content', '')}\n```\n"
        )
    return repo_context


# =====================================================
# Create system prompt
# =====================================================
def create_prompt(issue: dict, repo_context: str) -> str:
    """Creates a prompt with issue details and file context."""
    issue_title = issue.get("title", "Untitled Issue")
    issue_body = issue.get("body", "No description provided.")
    issue_number = issue.get("number", "N/A")

    system_prompt = f"""
You are Desolve AI — an AI code assistant that fixes repository issues based on given files.

### Issue #{issue_number}: {issue_title}
{issue_body}

### Repository Files
{repo_context}

Task:
- Suggest precise code changes or fixes.
- Use provided files only.
- Include reasoning and updated code snippets.
- Keep responses concise, professional, and formatted.
"""
    return system_prompt


# =====================================================
# Start interactive chat session
# =====================================================
def start_chat(system_prompt: str, model_name: str):
    """Starts CLI-based Gemini chat session."""
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat(history=[{"role": "user", "parts": system_prompt}])

    print("\n🤖 Desolve AI: Analysis complete. Let's solve this issue together!\n")

    while True:
        user_input = input("👨‍💻 You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("🔚 Exiting chat. Goodbye!")
            break

        try:
            response = chat.send_message(user_input)
            print("\n🤖 Desolve AI:\n")
            print(response.text)
            print("\n" + "-" * 90 + "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")


# =====================================================
# Main execution flow
# =====================================================
if __name__ == "__main__":
    load_env_and_configure()

    issues, files = load_repo_data(ISSUES_CSV, FILES_CSV)

    if len(issues) <= ROW_INDEX:
        print(f"❌ CSV has only {len(issues)} issues. Row {ROW_INDEX+1} not found.")
        exit(1)

    issue = issues[ROW_INDEX]
    repo_context = build_repo_context(files)
    system_prompt = create_prompt(issue, repo_context)

    print(f"\n📂 Loaded issue #{issue.get('number', 'N/A')} — “{issue.get('title', '')}”")
    start_chat(system_prompt, MODEL_NAME)
