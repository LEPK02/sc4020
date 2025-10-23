import subprocess
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "client" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)])
