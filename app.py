import os
from dotenv import load_dotenv
from clone import clone_repo
from file_contents import extract_files_to_csv
from issues import extract_issues

def run_pipeline(repo_url, clone_dir="cloned_repo"):

    # Determine local repo path
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(clone_dir, repo_name)

    # Step 1: Clone repo
    clone_repo(repo_url, clone_dir)

    # Step 2: Extract repository files
    files_csv = "repo_files_data.csv"
    print("\n📂 Extracting repository files...")
    extract_files_to_csv(repo_path, files_csv)

    # Step 3: Extract GitHub issues
    issues_csv = "repo_issues.csv"
    print("\n🐞 Extracting repository issues...")
    token = os.getenv("GITHUB_TOKEN")
    extract_issues(repo_url, output_file=issues_csv, token=token)

    print("\n✅ Pipeline completed successfully!")
    print(f"   - Files CSV: {files_csv}")
    print(f"   - Issues CSV: {issues_csv}")

def main():
    load_dotenv()  # Load environment variables (optional GitHub token)
    repo_url = input("🔗 Enter GitHub Repository URL: ").strip()

    if not repo_url:
        print("❌ No repository URL provided. Exiting.")
        return
    
    run_pipeline(repo_url)

if __name__ == "__main__":
    main()
