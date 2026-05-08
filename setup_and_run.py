import os
import sys
import subprocess
import platform
from pathlib import Path

def setup_and_run():
    """
    Cross-platform script to create a virtual environment,
    install dependencies, and start the FastAPI server.
    """
    project_root = Path(__file__).parent.resolve()
    venv_dir = project_root / "venv"
    
    # OS specifics
    is_windows = platform.system().lower() == "windows"
    
    # Define paths to pip and python inside the venv
    if is_windows:
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"

    # Step 1: Create Virtual Environment
    if not venv_dir.exists():
        print("🛠️  Creating virtual environment 'venv'...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            print("✅ Virtual environment created successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists.\n")

    # Step 2: Install dependencies
    requirements_path = project_root / "requirements.txt"
    if requirements_path.exists():
        print("📦 Installing/Updating dependencies from requirements.txt...")
        try:
            # Using the venv's pip ensures packages go into the venv
            subprocess.run([str(venv_pip), "install", "-r", str(requirements_path)], check=True)
            print("✅ Dependencies installed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        print("⚠️ requirements.txt not found. Skipping dependency installation.\n")

    # Step 3: Run the server
    api_path = project_root / "app" / "api.py"
    if api_path.exists():
        print("🚀 Starting the Image Search AI engine...")
        print("💡 Note: Running the server this way automatically uses the virtual environment.\n")
        print("-" * 50)
        try:
            # Running with venv_python is equivalent to activating it and running python
            subprocess.run([str(venv_python), str(api_path)], check=True)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped gracefully.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Server crashed: {e}")
    else:
        print(f"❌ Could not find the main app file at {api_path}")
        sys.exit(1)

if __name__ == "__main__":
    setup_and_run()
