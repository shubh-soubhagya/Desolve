import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# =====================================================
# Configuration variables
# =====================================================
MODEL_NAME = "llama3-70b-8192"   # You can also try: "mixtral-8x7b", "gemma-7b-it"
ISSUES_CSV = "repo_issues.csv"
FILES_CSV = "repo_files_data.csv"
ROW_INDEX = 3  # 4th issue (0-based index)

# =====================================================
# Load environment variables
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
# Build repository context
# =====================================================
def build_repo_context(files: list) -> str:
    """Combines all file data into a context string for the model."""
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
- Analyze and suggest precise code fixes.
- Use provided files only.
- Include reasoning and corrected code snippets.
- Keep responses professional and formatted.
"""
    return system_prompt


# =====================================================
# Interactive Groq chat
# =====================================================
def start_chat(system_prompt: str, model_name: str, client):
    """Starts CLI-based Groq chat session."""
    print("\n🤖 Desolve AI (Groq): Analysis complete. Let's solve this issue together!\n")

    history = [
        {"role": "system", "content": "You are Desolve AI — an expert AI code assistant."},
        {"role": "user", "content": system_prompt},
    ]

    while True:
        user_input = input("👨‍💻 You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("🔚 Exiting chat. Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=history,
                temperature=0.6,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            answer = response.choices[0].message.content
            print("\n🤖 Desolve AI:\n")
            print(answer)
            print("\n" + "-" * 90 + "\n")

            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"⚠️ Error: {e}")


# =====================================================
# Main execution flow
# =====================================================
if __name__ == "__main__":
    client = load_env_and_configure()

    issues, files = load_repo_data(ISSUES_CSV, FILES_CSV)

    if len(issues) <= ROW_INDEX:
        print(f"❌ CSV has only {len(issues)} issues. Row {ROW_INDEX+1} not found.")
        exit(1)

    issue = issues[ROW_INDEX]
    repo_context = build_repo_context(files)
    system_prompt = create_prompt(issue, repo_context)

    print(f"\n📂 Loaded issue #{issue.get('number', 'N/A')} — “{issue.get('title', '')}”")
    start_chat(system_prompt, MODEL_NAME, client)
