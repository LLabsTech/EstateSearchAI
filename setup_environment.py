# setup_environment.py
import os
import shutil

def setup_chroma_environment():
    """Set up ChromaDB environment"""
    chroma_dir = "./chroma_db"
    
    # Remove existing directory if it exists
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    
    # Create directory with proper permissions
    os.makedirs(chroma_dir, exist_ok=True)
    os.chmod(chroma_dir, 0o777)
    
    # Create required subdirectories
    for subdir in ['index', 'data']:
        path = os.path.join(chroma_dir, subdir)
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o777)

if __name__ == "__main__":
    setup_chroma_environment()
