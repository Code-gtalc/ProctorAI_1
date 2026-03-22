from __future__ import annotations

import os

# Suppress TensorFlow/oneDNN startup noise from transitive deps (e.g. MediaPipe/librosa).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from web_modules import create_app


if __name__ == "__main__":
    app = create_app()
    debug_mode = os.getenv("PG_DEBUG", "0").strip() in {"1", "true", "yes", "on"}
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=debug_mode,
        use_reloader=debug_mode,
    )
