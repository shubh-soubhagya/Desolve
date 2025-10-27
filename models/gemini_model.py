import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# =====================================================
# Load environment variables from .env file
# =====================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

# =====================================================
# Load CSVs safely using pandas
# =====================================================
issues = pd.read_csv(r"C:\Users\hp\Desktop\MinorProj\Desolve\repo_issues.csv", encoding='utf-8')
files = pd.read_csv(r"C:\Users\hp\Desktop\MinorProj\Desolve\repo_files_data.csv", encoding='utf-8')

if issues.empty:
    print("❌ No issues found in repo_issues.csv.")
    exit(1)

# Convert DataFrame to list of dicts (if needed)
issues = issues.to_dict(orient="records")
files = files.to_dict(orient="records")

# =====================================================
# Use first issue only
# =====================================================
first_issue = issues[3]
issue_title = first_issue.get("title", "Untitled Issue")
issue_body = first_issue.get("body", "No description provided.")
issue_number = first_issue.get("number", "N/A")

# =====================================================
# Build repository file context
# =====================================================
repo_context = ""
for file in files:
    repo_context += f"\n\n### File: {file.get('file_name', '')} ({file.get('file_path', '')})\n" \
                    f"```{file.get('file_extension', '')}\n{file.get('file_content', '')}\n```\n"

# =====================================================
# Prompt engineering for Gemini
# =====================================================
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

# =====================================================
# Start Gemini Chat
# =====================================================
model = genai.GenerativeModel(MODEL_NAME)
chat = model.start_chat(history=[{"role": "user", "parts": system_prompt}])

print(f"\n🤖 Desolve AI: I’ve analyzed issue #{issue_number} — “{issue_title}”. Let's solve it together!\n")

# =====================================================
# CLI Interaction
# =====================================================
while True:
    user_input = input("👨‍💻 You: ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        print("🔚 Exiting chat. Goodbye!")
        break

    response = chat.send_message(user_input)
    print("\n🤖 Desolve AI:\n")
    print(response.text)
    print("\n" + "-" * 90 + "\n")
