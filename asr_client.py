import logging
import multiprocessing as mp
import threading
import time
import uuid
from multiprocessing.connection import Connection

from asr_worker import worker_main

log = logging.getLogger("LiveTranslate.ASRClient")


class ASRClientError(RuntimeError):
    pass


class ASRWorkerError(ASRClientError):
    def __init__(self, error: dict):
        self.error = error
        self.recoverable = bool(error.get("recoverable", True))
        super().__init__(error.get("message", "ASR worker error"))


class ASRWorkerTimeout(ASRClientError):
    pass


class ASRWorkerExited(ASRClientError):
    pass


class ASRClient:
    """Main-process proxy for a single ASR worker process."""

    def __init__(
        self,
        config: dict,
        ready_timeout: float = 180.0,
        request_timeout: float = 120.0,
        shutdown_timeout: float = 5.0,
    ):
        self.config = dict(config)
        self.ready_timeout = ready_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self._ctx = mp.get_context("spawn")
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._lock = threading.RLock()
        self._status = "created"

    @property
    def status(self) -> str:
        if self._process is not None and self._process.exitcode is not None:
            if self._status not in ("failed", "stopping", "stopped"):
                self._status = "exited"
        return self._status

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self):
        with self._lock:
            if self._process is not None:
                return
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            name = f"ASRWorker-{self.config.get('engine_type', 'unknown')}"
            process = self._ctx.Process(
                target=worker_main,
                args=(child_conn, self.config),
                name=name,
            )
            process.daemon = True
            process.start()
            child_conn.close()
            self._conn = parent_conn
            self._process = process
            self._status = "starting"
            log.info(f"ASR worker started: pid={process.pid}, name={name}")

    def wait_ready(self, timeout: float | None = None):
        timeout = self.ready_timeout if timeout is None else timeout
        with self._lock:
            self._ensure_started()
            self._status = "loading"
            response = self._recv_response(timeout, expected_id=None)
            if not response.get("ok"):
                self._status = "failed"
                raise ASRWorkerError(response.get("error") or {})
            if response.get("type") != "ready":
                self._status = "failed"
                raise ASRClientError(
                    f"Unexpected ASR worker startup response: {response.get('type')}"
                )
            self._status = "ready"
            log.info(
                f"ASR worker ready: pid={self.pid}, "
                f"{response.get('payload') or {}}"
            )
            return response.get("payload")

    def transcribe(self, audio, word_timestamps: bool = False, **kwargs):
        payload = {"audio": audio, "word_timestamps": word_timestamps}
        payload.update(kwargs)
        response = self._request("transcribe", payload, timeout=self.request_timeout)
        return response.get("payload")

    def set_language(self, language: str):
        self._request(
            "set_language",
            {"language": language},
            timeout=min(10.0, self.request_timeout),
        )

    def set_input_padding(self, pad_seconds):
        self._request(
            "set_input_padding",
            {"pad_seconds": pad_seconds},
            timeout=min(10.0, self.request_timeout),
        )

    def shutdown(self):
        with self._lock:
            if self._process is None:
                return
            self._status = "stopping"
            process = self._process
            conn = self._conn
            if process.is_alive() and conn is not None:
                msg_id = uuid.uuid4().hex
                try:
                    conn.send({"id": msg_id, "type": "shutdown", "payload": {}})
                    if conn.poll(self.shutdown_timeout):
                        try:
                            conn.recv()
                        except EOFError:
                            pass
                except (BrokenPipeError, EOFError, OSError):
                    pass
            process.join(timeout=self.shutdown_timeout)
            if process.is_alive():
                log.warning(f"ASR worker did not exit, terminating pid={process.pid}")
                process.terminate()
                process.join(timeout=self.shutdown_timeout)
            self._close_handles()
            self._status = "stopped"
            log.info("ASR worker stopped")

    def terminate(self):
        with self._lock:
            if self._process is not None and self._process.is_alive():
                self._status = "failed"
                self._process.terminate()
                self._process.join(timeout=self.shutdown_timeout)
            self._close_handles()

    def _request(self, request_type: str, payload: dict, timeout: float):
        with self._lock:
            self._ensure_ready()
            msg_id = uuid.uuid4().hex
            try:
                self._conn.send(
                    {"id": msg_id, "type": request_type, "payload": payload}
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._status = "exited"
                raise ASRWorkerExited(f"ASR worker pipe closed: {exc}") from exc

            previous_status = self._status
            if request_type == "transcribe":
                self._status = "busy"
            try:
                response = self._recv_response(timeout, expected_id=msg_id)
            finally:
                if self._status == "busy":
                    self._status = previous_status

            if not response.get("ok"):
                raise ASRWorkerError(response.get("error") or {})
            return response

    def _recv_response(self, timeout: float, expected_id: str | None):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._status = "failed"
                self.terminate()
                raise ASRWorkerTimeout(
                    f"ASR worker response timed out after {timeout:g}s"
                )

            conn = self._conn
            if conn is not None and conn.poll(min(0.2, remaining)):
                try:
                    response = conn.recv()
                except EOFError as exc:
                    self._status = "exited"
                    raise ASRWorkerExited("ASR worker pipe closed") from exc
                if expected_id is None or response.get("id") == expected_id:
                    return response
                raise ASRClientError(
                    "ASR worker response id mismatch: "
                    f"expected={expected_id}, got={response.get('id')}"
                )

            process = self._process
            if process is not None and process.exitcode is not None:
                self._status = "exited"
                raise ASRWorkerExited(
                    f"ASR worker exited with code {process.exitcode}"
                )

    def _ensure_started(self):
        if self._process is None or self._conn is None:
            raise ASRClientError("ASR worker has not been started")

    def _ensure_ready(self):
        self._ensure_started()
        if self.status != "ready":
            raise ASRClientError(f"ASR worker is not ready: {self.status}")

    def _close_handles(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._process = None
