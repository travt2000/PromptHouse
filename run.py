"""Start the server: one process, one worker (it owns the GPU models), bound to localhost."""

import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    uvicorn.run("app.main:app", host=a.host, port=a.port, workers=1, reload=False, log_level="info")
