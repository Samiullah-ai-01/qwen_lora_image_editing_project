import os
import sys
import subprocess
import shutil
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

def ensure_directories():
    """Ensure all required project directories exist."""
    dirs = [
        ROOT_DIR / "models" / "base",
        ROOT_DIR / "models" / "loras",
        ROOT_DIR / "outputs" / "inference",
        ROOT_DIR / "outputs" / "logs",
        ROOT_DIR / "static",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def bootstrap():
    """The 'God Command' logic: Create venv, install deps, and build UI automatically."""
    ensure_directories()

    # 1. Check/Create .venv
    if not VENV_DIR.exists():
        print(">>> [INITIAL SETUP] Virtual environment not found. Building Imperial Workspace...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
            print(">>> [INITIAL SETUP] Virtual environment created successfully.")
        except Exception as e:
            print(f">>> [FATAL ERROR] Failed to create venv: {e}")
            sys.exit(1)

    # 2. Re-launch inside venv to handle dependencies correctly
    python_exe = get_python_executable()
    if not is_in_venv() and python_exe.exists():
        print(f">>> [BOOTSTRAP] Handing control to Virtual Environment: {python_exe}")
        os.environ["PYTHONPATH"] = str(SRC_DIR)
        os.execv(str(python_exe), [str(python_exe)] + sys.argv)

    # 3. WE ARE NOW INSIDE THE VENV
    # Verify/Install Requirements
    try:
        import diffusers
    except ImportError:
        print(">>> [DEPENDENCIES] Core ML libraries missing. Running installation (this may take a few minutes)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print(">>> [DEPENDENCIES] Installation completed.")

    # 4. Check/Build UI
    static_index = ROOT_DIR / "static" / "index.html"
    if not static_index.exists():
        print(">>> [UI] Production build missing. Building Imperial Studio UI...")
        try:
            # Check for Node.js
            try:
                subprocess.run(["node", "--version"], capture_output=True, check=True)
            except:
                print(">>> [WARNING] Node.js not found. Skipping UI build. Ensure 'static/' is populated manually or via Docker.")
                return

            # Proceed with build
            frontend_dir = SRC_DIR / "signforge" / "ui" / "frontend"
            if (frontend_dir / "package.json").exists():
                print(">>> [UI] Installing frontend assets...")
                subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True, check=True)
                print(">>> [UI] Forging production bundles...")
                subprocess.run(["npm", "run", "build"], cwd=frontend_dir, shell=True, check=True)
                print(">>> [UI] Forge complete.")
        except Exception as e:
            print(f">>> [UI ERROR] Failed to build frontend: {e}")

    # 5. Assistant Model Pre-cache
    try:
        from signforge.assistant.service import get_assistant
        print(">>> [ASSISTANT] Verifying Neural Assistant weights...")
        get_assistant()
    except Exception as e:
        print(f">>> [ASSISTANT] Note: Assistant pre-load skipped ({e})")

def main():
    # Setting up PYTHONPATH for internal modules
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from signforge.version import __version__, __codename__
    from signforge.core.device import get_device_manager
    
    device_mgr = get_device_manager()
    mem = device_mgr.get_memory_info()
    
    print("\n" + "="*50)
    print(f"       SIGNFORGE STUDIO v{__version__} '{__codename__}'")
    print("="*50)
    print(f"Workspace: {ROOT_DIR}")
    print(f"Engine:    {device_mgr.device} ({device_mgr.dtype})")
    if device_mgr.is_cuda_available:
        print(f"VRAM:      {mem.get('total_gb', 0):.1f} GB Total / {mem.get('free_gb', 0):.1f} GB Free")
    print("-" * 50)
    
    # Final check: Ensure we can at least find our core server
    try:
        from signforge.server.app import run_server
        print(">>> Initializing Imperial Neural Links...")
        run_server()
    except Exception as e:
        print(f">>> [SYSTEM ERROR] Forge ignition failed: {e}")
        print("\nPossible fix: Run setup manually or check your installations.")

if __name__ == "__main__":
    if "--no-bootstrap" not in sys.argv:
        bootstrap()
    main()
