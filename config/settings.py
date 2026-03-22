from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

RAG_ENGINE_MODE = os.getenv("RAG_ENGINE_MODE", "core")

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)


def _first(keys: Iterable[str], default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _get_int(keys: Iterable[str], default: int) -> int:
    value = _first(keys)
    return int(value) if value is not None else default


def _get_float(keys: Iterable[str], default: float) -> float:
    value = _first(keys)
    return float(value) if value is not None else default


def _get_csv(keys: Iterable[str], default: tuple[str, ...]) -> tuple[str, ...]:
    value = _first(keys)
    if value is None:
        return default
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or default


def _build_url(protocol: str, host: str, port: int, path: str = "") -> str:
    normalized_path = path if path.startswith("/") or not path else f"/{path}"
    return f"{protocol}://{host}:{port}{normalized_path}"


def _host_from_url(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname or fallback


@dataclass(frozen=True)
class RuntimeSettings:
    app_env: str
    log_level: str


@dataclass(frozen=True)
class FrontendSettings:
    host: str
    port: int
    protocol: str
    public_host: str
    transport_mode: str

    @property
    def public_url(self) -> str:
        return _build_url(self.protocol, self.public_host, self.port)


@dataclass(frozen=True)
class BackendSettings:
    host: str
    port: int
    protocol: str
    public_host: str
    ws_protocol: str
    ws_path: str
    api_prefix: str
    allowed_origins: tuple[str, ...]

    @property
    def public_url(self) -> str:
        return _build_url(self.protocol, self.public_host, self.port, self.api_prefix)

    @property
    def ws_url(self) -> str:
        return _build_url(self.ws_protocol, self.public_host, self.port, self.ws_path)


@dataclass(frozen=True)
class RagSettings:
    engine_mode: str
    max_retries: int
    hallucination_threshold: float
    top_k_docs: int
    faiss_index_path: str
    debug_step_delay_ms: int


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    model_name: str
    base_url: str
    temperature: float


@dataclass(frozen=True)
class SystemSettings:
    runtime: RuntimeSettings
    frontend: FrontendSettings
    backend: BackendSettings
    rag: RagSettings
    llm: LLMSettings


def load_settings() -> SystemSettings:
    runtime = RuntimeSettings(
        app_env=_first(("APP_ENV",), "development") or "development",
        log_level=(_first(("LOG_LEVEL",), "INFO") or "INFO").upper(),
    )

    frontend_host = _first(("FRONTEND_HOST",), "127.0.0.1") or "127.0.0.1"
    frontend_port = _get_int(("FRONTEND_PORT",), 5173)
    frontend_protocol = _first(("FRONTEND_PROTOCOL",), "http") or "http"
    frontend_public_url = _first(("FRONTEND_PUBLIC_URL",))
    frontend_public_host = (
        _host_from_url(frontend_public_url, frontend_host)
        if frontend_public_url
        else frontend_host
    )

    frontend = FrontendSettings(
        host=frontend_host,
        port=frontend_port,
        protocol=frontend_protocol,
        public_host=frontend_public_host,
        transport_mode=_first(("FRONTEND_RAG_TRANSPORT",), "websocket") or "websocket",
    )

    backend_host = _first(("BACKEND_HOST",), "0.0.0.0") or "0.0.0.0"
    backend_port = _get_int(("BACKEND_PORT",), 8000)
    backend_protocol = _first(("BACKEND_PROTOCOL",), "http") or "http"
    backend_ws_protocol = _first(("BACKEND_WS_PROTOCOL",), "ws") or "ws"
    backend_ws_path = _first(("BACKEND_WS_PATH",), "/ws/rag") or "/ws/rag"
    backend_api_prefix = _first(("BACKEND_API_PREFIX",), "") or ""
    backend_public_url = _first(("BACKEND_PUBLIC_URL",))
    default_backend_public_host = (
        "127.0.0.1" if backend_host == "0.0.0.0" else backend_host
    )
    backend_public_host = (
        _host_from_url(backend_public_url, default_backend_public_host)
        if backend_public_url
        else default_backend_public_host
    )

    default_allowed_origins = tuple(
        dict.fromkeys(
            (
                frontend.public_url,
                _build_url(frontend_protocol, "localhost", frontend_port),
                _build_url(frontend_protocol, "127.0.0.1", frontend_port),
            )
        )
    )

    backend = BackendSettings(
        host=backend_host,
        port=backend_port,
        protocol=backend_protocol,
        public_host=backend_public_host,
        ws_protocol=backend_ws_protocol,
        ws_path=backend_ws_path,
        api_prefix=backend_api_prefix,
        allowed_origins=_get_csv(
            ("ALLOWED_ORIGINS", "BACKEND_ALLOWED_ORIGINS"),
            default_allowed_origins,
        ),
    )

    rag = RagSettings(
        engine_mode=(
            _first(("RAG_ENGINE_MODE", "BACKEND_ENGINE_MODE"), "core") or "core"
        ).lower(),
        max_retries=_get_int(("RAG_MAX_RETRIES",), 2),
        hallucination_threshold=_get_float(("RAG_HALLUCINATION_THRESHOLD",), 0.7),
        top_k_docs=_get_int(("RAG_TOP_K_DOCS",), 5),
        faiss_index_path=_first(
            ("RAG_FAISS_INDEX_PATH", "FAISS_INDEX_PATH"),
            "./data/faiss_index",
        )
        or "./data/faiss_index",
        debug_step_delay_ms=_get_int(
            ("DEBUG_STEP_DELAY_MS", "BACKEND_DEBUG_STEP_DELAY_MS"),
            250,
        ),
    )

    llm = LLMSettings(
        api_key=_first(("LLM_API_KEY",)),
        model_name=_first(("LLM_MODEL",), "qwen-turbo") or "qwen-turbo",
        base_url=(
            _first(
                ("LLM_BASE_URL",), "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        temperature=_get_float(("LLM_TEMPERATURE",), 0.0),
    )

    return SystemSettings(
        runtime=runtime,
        frontend=frontend,
        backend=backend,
        rag=rag,
        llm=llm,
    )


settings = load_settings()
