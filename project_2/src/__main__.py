import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.append(str(root))

    app_path = root / "client" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)], cwd=root)
