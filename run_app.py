import os
import subprocess
import sys

# Set project root so all module imports work
project_root = os.path.dirname(os.path.abspath(__file__))
env = os.environ.copy()
env["PYTHONPATH"] = project_root

# Load .env manually so GROQ_API_KEY is available
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))
env["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

print("Starting AgentRAG...")
print(f"Project root: {project_root}")

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "chat_app.py"],
    env=env,
    cwd=project_root
)