import os
import sys
import subprocess
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.absolute()
SRC_DIR = ROOT_DIR / "src"
VENV_DIR = ROOT_DIR / ".venv"

def get_python_executable():
    """Return the path to the python executable in the virtual environment."""
    if os.name == 'nt':
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"

def is_in_venv():
    """Check if the current process is running from the virtual environment."""
    return sys.prefix == str(VENV_DIR)

def bootstrap():
    """Ensure the environment is ready and re-run this script from the venv if needed."""
    # 1. Check for .venv
    if not VENV_DIR.exists():
        print(">>> Virtual environment not found. Starting initial setup...")
        setup_script = "scripts/setup_dev.ps1" if os.name == 'nt' else "scripts/setup_dev.sh"
        cmd = ["powershell", "-File", setup_script] if os.name == 'nt' else ["bash", setup_script]
        
        try:
            subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        except subprocess.CalledProcessError:
            print(">>> ERROR: Setup script failed. Please check the logs.")
            sys.exit(1)

    # 2. Check if we need to switch to the venv interpreter
    python_exe = get_python_executable()
    if not is_in_venv() and python_exe.exists():
        print(f">>> Switching to virtual environment: {python_exe}")
        # Re-run this script with the venv python
        os.environ["PYTHONPATH"] = str(SRC_DIR)
        os.execv(str(python_exe), [str(python_exe)] + sys.argv)

def check_frontend():
    """Ensure the frontend is built."""
    static_dir = ROOT_DIR / "static"
    if not (static_dir / "index.html").exists():
        print(">>> Frontend build missing. Building UI...")
        try:
            # We are already in venv if bootstrap ran
            subprocess.run([sys.executable, "-m", "signforge.ui.build"], cwd=ROOT_DIR, check=True)
        except Exception as e:
            print(f">>> WARNING: Frontend build failed ({e}). The server will start but UI might be missing.")

def main():
    # Setting up PYTHONPATH just in case
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    print("\n--- SignForge Local Studio ---")
    print(f"Working Directory: {ROOT_DIR}")
    
    # Final check for frontend
    check_frontend()

    # Start the server
    print(">>> Starting SignForge Server...")
    try:
        from signforge.server.app import run_server
        run_server()
    except ImportError as e:
        print(f">>> ERROR: Could not import SignForge components. Ensure you ran the setup script. ({e})")
    except KeyboardInterrupt:
        print("\n>>> Server stopped by user.")

if __name__ == "__main__":
    if "--no-bootstrap" not in sys.argv:
        bootstrap()
    main()
