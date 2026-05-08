import os
import shutil
from pathlib import Path

def clean_repo():
    """
    Clears all generated data, models, virtual environments, and caches
    to reset the repository to a freshly downloaded state.
    """
    project_root = Path(__file__).parent.resolve()
    
    # Directories to delete completely
    dirs_to_delete = [
        "data",
        "models",
        "venv",
        ".venv"
    ]
    
    print("🧹 Cleaning repository...\n")
    
    # Delete main generated directories
    for dir_name in dirs_to_delete:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"Removing directory: {dir_path.name}/")
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"  ❌ Failed to remove {dir_name}/: {e}")
                
    # Recursively find and delete __pycache__ and .pyc files
    pycache_count = 0
    for path in project_root.rglob("__pycache__"):
        if path.is_dir():
            try:
                shutil.rmtree(path)
                pycache_count += 1
            except Exception as e:
                pass
                
    pyc_count = 0
    for path in project_root.rglob("*.pyc"):
        if path.is_file():
            try:
                path.unlink()
                pyc_count += 1
            except Exception as e:
                pass
                
    if pycache_count > 0 or pyc_count > 0:
        print(f"Removed {pycache_count} __pycache__ directories and {pyc_count} .pyc files.")
        
    print("\n✨ Repository is now completely clean, like a fresh clone!")

if __name__ == "__main__":
    confirm = input("⚠️ This will delete all downloaded models, uploaded images, and virtual environments. Are you sure? (y/N): ")
    if confirm.lower() == 'y':
        clean_repo()
    else:
        print("Operation cancelled.")
