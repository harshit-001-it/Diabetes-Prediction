"""
run.py — Single-command launcher  (ONLY file at the project root)
==================================================================
1. Trains all models if models/best_model.pkl doesn't exist yet.
2. Launches the Flask server in its OWN console window — so it
   keeps running even after you close this terminal or Chrome.
3. Opens your browser at http://127.0.0.1:5000 automatically.

Usage:
    python run.py          ← start everything
    python run.py --train  ← force re-train even if model exists
    python run.py --stop   ← stop the running Flask server
"""

import os
import sys
import time
import subprocess
import webbrowser

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
PIPELINE    = os.path.join(ROOT, "src", "pipeline.py")
APP_SCRIPT  = os.path.join(ROOT, "src", "app.py")
MODEL_FILE  = os.path.join(ROOT, "models", "best_model.pkl")
PID_FILE    = os.path.join(ROOT, "models", "flask.pid")
FLASK_URL   = "http://127.0.0.1:5000"

# Windows flag: open a new console window for Flask
CREATE_NEW_CONSOLE = 0x00000010


def train():
    """Run the ML pipeline."""
    print("\n[run.py] 🚀  Training models — this takes ~30 seconds…\n")
    result = subprocess.run([sys.executable, PIPELINE])
    if result.returncode != 0:
        sys.exit("[run.py] ❌  Training failed. Fix the errors above and retry.")
    print("\n[run.py] ✅  Training complete.\n")


def start_flask():
    """
    Launch Flask in a SEPARATE console window so it stays alive
    even after this terminal or Chrome is closed.
    """
    proc = subprocess.Popen(
        [sys.executable, APP_SCRIPT],
        creationflags=CREATE_NEW_CONSOLE,   # own window — independent process
    )
    # Save PID so --stop can kill it later
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    return proc


def stop_flask():
    """Kill the saved Flask process (Windows)."""
    if not os.path.exists(PID_FILE):
        print("[run.py] No running Flask server found (no models/flask.pid).")
        return
    with open(PID_FILE) as f:
        pid = f.read().strip()
    try:
        subprocess.run(["taskkill", "/PID", pid, "/F"], check=True)
        os.remove(PID_FILE)
        print(f"[run.py] ✅  Flask server (PID {pid}) stopped.")
    except Exception as e:
        print(f"[run.py] ⚠️  Could not stop process: {e}")


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if "--stop" in sys.argv:
        stop_flask()
        sys.exit(0)

    force_train = "--train" in sys.argv

    # Step 1 — Train if needed
    if force_train or not os.path.exists(MODEL_FILE):
        train()
    else:
        print("[run.py] ✅  Model already trained — skipping training.")
        print("             (Use 'python run.py --train' to force re-train)")

    # Step 2 — Stop any old Flask instance
    if os.path.exists(PID_FILE):
        stop_flask()
        time.sleep(1)

    # Step 3 — Launch Flask in its own window
    proc = start_flask()
    time.sleep(2)   # give Flask time to start

    # Step 4 — Open browser
    webbrowser.open(FLASK_URL)

    print()
    print("=" * 55)
    print("  🌐  Web App  :  http://127.0.0.1:5000")
    print(f"  ⚙️   Flask PID :  {proc.pid}  (models/flask.pid)")
    print()
    print("  ✅  You can close THIS terminal window safely.")
    print("      Flask is running in its own separate window.")
    print()
    print("  🛑  To stop Flask later, run:")
    print("         python run.py --stop")
    print("=" * 55)
