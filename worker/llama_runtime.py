from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from worker.errors import StaleAttempt


DEFAULT_MODEL_REPO = "tencent/Hy-MT2-7B-GGUF"
DEFAULT_MODEL_FILE = "Hy-MT2-7B-Q4_K_M.gguf"
DEFAULT_MODEL_SHA256 = "9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b"
DEFAULT_LLAMA_REPO = "ggml-org/llama.cpp"
DEFAULT_LLAMA_FALLBACK = "b12503"


class LlamaRuntime:
    """Downloads a llama.cpp release binary and a GGUF model, then runs llama-server locally."""

    def __init__(self, heartbeat=None):
        self.cache = Path(os.getenv("LLAMA_CACHE", ".cache/offload-translate")).resolve()
        self.bin_dir = self.cache / "llamacpp"
        self.model_dir = self.cache / "models"
        self.model_repo = os.getenv("MODEL_REPO", DEFAULT_MODEL_REPO)
        self.model_file = os.getenv("MODEL_FILE", DEFAULT_MODEL_FILE)
        configured_sha = os.getenv("MODEL_SHA256", "").strip().lower()
        self.model_sha256 = configured_sha or (
            DEFAULT_MODEL_SHA256
            if self.model_repo == DEFAULT_MODEL_REPO and self.model_file == DEFAULT_MODEL_FILE
            else ""
        )
        self.llama_repo = os.getenv("LLAMA_REPO", DEFAULT_LLAMA_REPO)
        self.llama_version = os.getenv("LLAMA_VERSION", "").strip()
        self.endpoint = (os.getenv("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        self.context = int(os.getenv("MODEL_CONTEXT", "8192"))
        self.threads = int(os.getenv("MODEL_THREADS", str(max(1, min(4, os.cpu_count() or 4)))))
        self.port = int(os.getenv("LLAMA_PORT", "18080"))
        self.timeout = int(os.getenv("MODEL_REQUEST_TIMEOUT", "900"))
        self.heartbeat = heartbeat
        self.model_path = self.model_dir / self.model_file
        self.server_path: Path | None = None
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    def _download_heartbeat(self, label: str, detail: str) -> None:
        if not self.heartbeat:
            return
        try:
            self.heartbeat(f"{label}：{detail}")
        except StaleAttempt:
            raise
        except Exception as exc:
            # A temporary storage outage should not restart a multi-gigabyte download.
            print(f"Unable to persist download heartbeat: {exc}", flush=True)

    @staticmethod
    def _download(url: str, destination: Path, *, label: str, progress_callback=None) -> None:
        """Download with HTTP Range resume; large GGUF transfers should survive transient drops."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(6):
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "OffloadTranslate/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    if offset and not append:
                        offset = 0
                    mode = "ab" if append else "wb"
                    remaining = int(response.headers.get("Content-Length", "0") or 0)
                    content_range = response.headers.get("Content-Range", "")
                    expected_total = 0
                    if content_range and "/" in content_range:
                        try:
                            expected_total = int(content_range.rsplit("/", 1)[1])
                        except ValueError:
                            expected_total = 0
                    elif remaining:
                        expected_total = offset + remaining
                    downloaded = offset
                    next_report = 10
                    last_callback_bytes = downloaded
                    with temporary.open(mode) as output:
                        while True:
                            chunk = response.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            downloaded += len(chunk)
                            if expected_total:
                                percent = downloaded * 100 // expected_total
                                if percent >= next_report:
                                    detail = f"{percent}% ({downloaded}/{expected_total} bytes)"
                                    print(f"{label}: {detail}", flush=True)
                                    if progress_callback:
                                        progress_callback(label, detail)
                                    next_report = min(100, percent + 10)
                                    last_callback_bytes = downloaded
                            elif progress_callback and downloaded - last_callback_bytes >= 256 * 1024 * 1024:
                                detail = f"已下载 {downloaded} bytes"
                                progress_callback(label, detail)
                                last_callback_bytes = downloaded
                size = temporary.stat().st_size if temporary.exists() else 0
                if size == 0:
                    raise RuntimeError(f"{label}: downloaded file is empty")
                if expected_total and size != expected_total:
                    raise RuntimeError(f"{label}: incomplete download {size}/{expected_total} bytes")
                temporary.replace(destination)
                return
            except StaleAttempt:
                raise
            except Exception as exc:
                last_error = exc
                print(f"{label}: download attempt {attempt + 1}/6 failed: {exc}", flush=True)
                if attempt < 5:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{label}: download failed after retries: {last_error}") from last_error

    def _release_asset(self) -> tuple[str, str]:
        if self.llama_version:
            tag = self.llama_version
        else:
            url = f"https://api.github.com/repos/{self.llama_repo}/releases"
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "OffloadTranslate/1.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    releases = json.load(response)
                if not releases:
                    raise RuntimeError("GitHub returned an empty llama.cpp release list")
                tag = str(releases[0]["tag_name"])
            except Exception as exc:
                print(f"Could not resolve the latest llama.cpp release: {exc}")
                tag = DEFAULT_LLAMA_FALLBACK
        asset = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
        return tag, asset

    def _find_server(self) -> Path | None:
        if self.bin_dir.exists():
            for path in self.bin_dir.rglob("llama-server"):
                if path.is_file():
                    return path.resolve()
        found = shutil.which("llama-server")
        return Path(found).resolve() if found else None

    def install_llama(self) -> Path:
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError("The packaged llama.cpp release currently targets Linux x86_64 Actions runners")
        tag, asset = self._release_asset()
        self.bin_dir = self.cache / "llamacpp" / tag
        existing = self._find_server()
        if existing:
            self.server_path = existing
            return existing
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        archive = self.bin_dir / asset
        url = f"https://github.com/{self.llama_repo}/releases/download/{tag}/{asset}"
        print(f"Installing llama.cpp {tag} from {url}", flush=True)
        self._download(
            url, archive, label="llama.cpp binary", progress_callback=self._download_heartbeat
        )
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=self.bin_dir, filter="data")
        finally:
            archive.unlink(missing_ok=True)
        server = self._find_server()
        if server is None:
            raise RuntimeError(f"llama-server not found in {asset}; release asset naming may have changed")
        server.chmod(server.stat().st_mode | 0o111)
        self.server_path = server
        print(f"llama-server ready: {server}", flush=True)
        return server

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_model(self) -> None:
        if not self.model_sha256:
            return
        actual = self._sha256(self.model_path)
        if actual.lower() != self.model_sha256:
            self.model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"GGUF SHA256 mismatch: expected {self.model_sha256}, got {actual}; cached/downloaded file removed"
            )
        print(f"GGUF SHA256 verified: {actual}", flush=True)

    def install_model(self) -> Path:
        if self.model_path.is_file() and self.model_path.stat().st_size > 100_000_000:
            print(f"Using cached model: {self.model_path}", flush=True)
            try:
                self._verify_model()
            except RuntimeError as exc:
                # A cache may outlive a model revision. The failed file has been removed;
                # fetch the configured model again instead of failing on a stale cache hit.
                print(f"Cached GGUF rejected: {exc}; downloading a fresh copy", flush=True)
            else:
                return self.model_path
        self.model_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.endpoint}/{self.model_repo}/resolve/main/{self.model_file}?download=true"
        print(f"Downloading GGUF model: {self.model_repo}/{self.model_file}", flush=True)
        self._download(
            url, self.model_path, label="Hunyuan 7B Q4_K_M model", progress_callback=self._download_heartbeat
        )
        self._verify_model()
        return self.model_path

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as response:
                return response.status == 200
        except Exception:
            return False

    def _port_open(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.4)
            try:
                sock.connect(("127.0.0.1", self.port))
                return True
            except OSError:
                return False

    def start(self):
        if self._healthy():
            print("An existing local llama-server is healthy", flush=True)
            return self
        if self._port_open():
            raise RuntimeError(f"Local llama-server port {self.port} is already occupied")
        if self.heartbeat:
            self.heartbeat("正在准备 llama.cpp 和 Hunyuan 7B 模型")
        binary = self.install_llama()
        model = self.install_model()
        self.log_handle = (self.cache / "llama-server.log").open("a", encoding="utf-8")
        command = [
            str(binary), "-m", str(model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.context), "-t", str(self.threads),
            "--parallel", "1",
        ]
        print("Starting local llama-server: " + " ".join(command), flush=True)
        process_env = os.environ.copy()
        library_dirs = {str(binary.parent)}
        for library in self.bin_dir.rglob("lib*.so*"):
            library_dirs.add(str(library.parent))
        existing_ld = process_env.get("LD_LIBRARY_PATH", "")
        process_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*sorted(library_dirs), *([existing_ld] if existing_ld else [])]
        )
        self.process = subprocess.Popen(
            command, stdout=self.log_handle, stderr=subprocess.STDOUT, env=process_env
        )
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = ""
                try:
                    text = (self.cache / "llama-server.log").read_text(encoding="utf-8", errors="replace")
                    tail = "\n".join(text.splitlines()[-50:])
                except OSError:
                    pass
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}. Log tail:\n{tail}")
            if self._healthy():
                print("Hunyuan 7B llama-server is healthy on 127.0.0.1", flush=True)
                if self.heartbeat:
                    self.heartbeat("模型服务已就绪")
                return self
            time.sleep(1)
        self.close()
        raise TimeoutError("llama-server did not become healthy within 240 seconds")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
