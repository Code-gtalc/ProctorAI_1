from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "react_dashboard_app"


def _stream_output(prefix: str, pipe) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        print(f"[{prefix}] {line.rstrip()}")


def main() -> int:
    backend_cmd = [sys.executable, "web_enrollment_app.py"]
    frontend_cmd = ["npm.cmd", "run", "dev"]
    backend_env = os.environ.copy()
    backend_env.setdefault("PG_DEBUG", "0")
    backend_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    backend_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    backend_env.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    backend_env.setdefault("GLOG_minloglevel", "2")

    backend = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT),
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    frontend = subprocess.Popen(
        frontend_cmd,
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
    )

    threads = [
        threading.Thread(target=_stream_output, args=("BACKEND", backend.stdout), daemon=True),
        threading.Thread(target=_stream_output, args=("FRONTEND", frontend.stdout), daemon=True),
    ]
    for t in threads:
        t.start()

    print("Both services started.")
    print("Backend:  http://127.0.0.1:5000")
    print("Frontend: check Vite output (usually http://127.0.0.1:5173)")
    print("Press Ctrl+C to stop both.")

    try:
        while True:
            backend_rc = backend.poll()
            frontend_rc = frontend.poll()
            if backend_rc is not None:
                print(f"Backend exited with code {backend_rc}")
                break
            if frontend_rc is not None:
                print(f"Frontend exited with code {frontend_rc}")
                break
            signal.pause() if os.name != "nt" else threading.Event().wait(0.25)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in (backend, frontend):
            if proc.poll() is None:
                proc.terminate()
        for proc in (backend, frontend):
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
