"""A stand-in for ``llama-server`` used by the tests.

Accepts the real command line (``-m``, ``--mmproj``, ``--port`` ...), reports 503 on
``/health`` for a moment (as the real server does while loading), then answers
``/v1/chat/completions`` with a valid record for the analysis kind it is asked
about (recognised by its system prompt). Special behaviours are triggered by the
``FAKE_LLAMA_MODE`` environment variable: ``crash`` (exit immediately), ``bad_json``
(always invalid output), ``no_schema`` (HTTP 400 for any ``response_format``).
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.style.guides import GUIDES  # noqa: E402
from app.style.kinds import KINDS  # noqa: E402
from app.style.prompts import system_prompt as prompts_system  # noqa: E402
from app.style.schema import sample_record  # noqa: E402

MODE = os.environ.get("FAKE_LLAMA_MODE", "")
READY_AT = time.monotonic() + 0.4
SEEN = os.environ.get("FAKE_LLAMA_SEEN")  # optional file that receives one line per request
ARGV = os.environ.get("FAKE_LLAMA_ARGV")  # optional file that receives the command line as JSON


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200 if time.monotonic() >= READY_AT else 503, {"status": "ok"})
        else:
            self._send(404, {})

    def _guide(self, body, guide_kind):
        """Text-only guide request: answer with every section heading, or per FAKE_LLAMA_MODE."""
        if SEEN:
            with open(SEEN, "a") as f:
                f.write(json.dumps({"guide": guide_kind, "user": body["messages"][-1]["content"],
                                    "max_tokens": body.get("max_tokens")}) + "\n")
        if MODE == "guide_slow":
            time.sleep(30)
        if MODE == "guide_error":
            return self._send(500, {"error": "boom"})
        spec = GUIDES[guide_kind]
        sections = spec.sections[:-1] if MODE == "guide_short" else spec.sections
        text = "<think>planning</think># A Title The Model Added\n\n" + "\n\n".join(f"## {s}\nFake text for {s}." for s in sections)
        self._send(200, {"choices": [{"message": {"content": text},
                                      "finish_reason": "length" if MODE == "guide_short" else "stop"}]})

    def _prompts(self, body):
        """Prompt-writing request: a numbered list with as many prompts as asked for (with the usual noise)."""
        user = body["messages"][-1]["content"]
        if SEEN:
            with open(SEEN, "a") as f:
                f.write(json.dumps({"prompts": True, "user": user, "max_tokens": body.get("max_tokens"),
                                    "temperature": body.get("temperature")}) + "\n")
        if MODE == "guide_slow":
            time.sleep(30)
        if MODE == "prompts_empty":
            return self._send(200, {"choices": [{"message": {"content": "<think>x</think>Sorry."}, "finish_reason": "stop"}]})
        n = int(re.search(r"Write (\d+) prompt", user).group(1))
        items = [f"{i}. **Idea {i}:** A medium shot of the subject in scene number {i}, soft window light, warm tones."
                 for i in range(1, n + 1)]
        text = "<think>planning</think>Here are your prompts:\n\n" + "\n".join(items)
        self._send(200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["messages"][0]["content"] == prompts_system():
            return self._prompts(body)
        guide_kind = next((k for k, g in GUIDES.items() if g.system_prompt == body["messages"][0]["content"]), None)
        if guide_kind:
            return self._guide(body, guide_kind)
        content = body["messages"][-1]["content"]
        has_image = any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in content)
        fmt = (body.get("response_format") or {}).get("type", "none")
        system = body["messages"][0]["content"]
        kind = next((k for k in KINDS.values() if k.system_prompt == system), None)
        if SEEN:
            with open(SEEN, "a") as f:
                f.write(json.dumps({"image": has_image, "format": fmt, "kind": kind and kind.key,
                                    "kwargs": body.get("chat_template_kwargs")}) + "\n")
        if MODE == "no_schema" and "response_format" in body:
            return self._send(400, {"error": "response_format not supported"})
        if not has_image:
            return self._send(400, {"error": "no image"})
        if kind is None:
            return self._send(400, {"error": "unrecognised system prompt"})
        text = "this is not json" if MODE == "bad_json" else "<think>hmm</think>```json\n" + json.dumps(sample_record(kind.schema)) + "\n```"
        self._send(200, {"choices": [{"message": {"content": text}}]})


def main():
    args = sys.argv[1:]
    if MODE == "crash":
        print("error: unknown model architecture", file=sys.stderr)
        sys.exit(3)
    if ARGV:
        with open(ARGV, "w") as f:
            json.dump(args, f)
    port = int(args[args.index("--port") + 1])
    print("fake llama-server listening", port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
