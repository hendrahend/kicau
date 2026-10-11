"""Tiny offline web server for Kicau. Standard library only.

Routes:
  GET  /              -> web UI
  GET  /api/health    -> {"ok": true, "species": 6522}
  POST /api/identify  -> multipart form with field "audio" -> identification JSON

Everything runs on-device: the uploaded audio never leaves this machine.
"""

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

_identifier = None


def get_identifier():
    global _identifier
    if _identifier is None:
        from .identify import BirdIdentifier

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _identifier = BirdIdentifier(
            model_path=os.path.join(base, "model", "audio-model-fp16.tflite"),
            labels_path=os.path.join(base, "kicau", "labels_id.txt"),
        )
    return _identifier


def parse_multipart_file(body: bytes, content_type: str):
    """Minimal multipart parser: returns (filename, file_bytes) of the first file part."""
    boundary = content_type.split("boundary=")[1].encode()
    parts = body.split(b"--" + boundary)
    for part in parts:
        if b"filename=" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        data = data.rsplit(b"\r\n", 1)[0]
        filename = header.split(b'filename="')[1].split(b'"')[0].decode("utf-8", "replace")
        return filename, data
    raise ValueError("no file part found in upload")


class Handler(BaseHTTPRequestHandler):
    server_version = "Kicau/0.1.0"

    def log_message(self, *args):
        pass  # keep the terminal clean

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            try:
                ident = get_identifier()
                self._json({"ok": True, "species": len(ident.labels), "offline": True})
            except Exception as e:  # noqa: BLE001 - surfaced as JSON for the UI
                self._json({"ok": False, "error": str(e)}, status=500)
            return
        if path in ("/", "/index.html"):
            fpath = os.path.join(WEB_DIR, "index.html")
            with open(fpath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/identify":
            self.send_error(404)
            return
        lang = parse_qs(parsed.query).get("lang", ["en"])[0]
        if lang not in ("en", "id"):
            lang = "en"
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._json({"error": "empty or too large upload (max 50 MB)"}, status=400)
            return
        try:
            body = self.rfile.read(length)
            filename, data = parse_multipart_file(body, self.headers.get("Content-Type", ""))
        except Exception as e:  # noqa: BLE001 - bad upload, tell the UI
            self._json({"error": f"could not read upload: {e}"}, status=400)
            return
        suffix = os.path.splitext(filename)[1].lower() or ".webm"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(data)
            tmp.close()
            result = get_identifier().identify_file(tmp.name, lang=lang)
            result["file"] = filename
            result["lang"] = lang
            self._json(result)
        except Exception as e:  # noqa: BLE001 - surfaced as JSON for the UI
            self._json({"error": str(e)}, status=500)
        finally:
            os.unlink(tmp.name)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Kicau offline bird-call server.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    # Fail fast with a clear message if the model is missing.
    get_identifier()
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Kicau berjalan di http://{args.host}:{args.port}  (100% offline)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
