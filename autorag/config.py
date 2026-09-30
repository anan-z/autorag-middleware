"""Configuration loading for AutoRAG Middleware."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

try:
    import platformdirs
except ImportError:  # pragma: no cover — platformdirs is a declared dependency
    import os
    import sys

    class _PD:
        @staticmethod
        def user_data_dir(app: str) -> str:
            if os.name == "nt":
                base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            elif sys.platform == "darwin":
                base = str(Path.home() / "Library" / "Application Support")
            else:
                base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
            return str(Path(base) / app)

        @staticmethod
        def user_config_dir(app: str) -> str:
            if os.name == "nt":
                base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            elif sys.platform == "darwin":
                base = str(Path.home() / "Library" / "Application Support")
            else:
                base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
            return str(Path(base) / app)

    platformdirs = _PD()  # type: ignore


class LLMBackend(BaseModel):
    api_base: str
    api_key: str = "not-needed"
    model: str = "default"


class ExtractionConfig(BaseModel):
    enabled: bool = True
    use_spacy: bool = False
    min_confidence: float = 0.55
    mode: Literal["heuristic", "llm", "hybrid"] = "hybrid"
    llm_every_n_turns: int = 1
    llm_max_tokens: int = 400
    llm_temperature: float = 0.1


class ValidationConfig(BaseModel):
    enabled: bool = True
    # Memory commit conflicts (fact write path)
    on_conflict: Literal["off", "flag", "reconcile"] = "reconcile"
    # Post-generation draft check against memory
    # off | flag | soft | hard | reconcile
    response_policy: Literal["off", "flag", "soft", "hard", "reconcile"] = "hard"
    max_retries: int = 1
    # Use same-model JSON claim extract for domain-neutral reality check
    use_llm_claims: bool = True
    # Optional pattern packs: e.g. ["rp"] for narrative stress-test patterns
    extra_patterns: list[str] = Field(default_factory=list)


class AutoRAGConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    database_path: Path | None = None
    backends: dict[str, LLMBackend] = Field(default_factory=dict)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    context_window: int = 8000
    max_injected_tokens: int = 1500
    default_conversation_id: str = "default"

    def resolved_db_path(self) -> Path:
        if self.database_path is not None:
            return Path(self.database_path)
        data_dir = Path(platformdirs.user_data_dir("autorag-middleware"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "state.db"

    @classmethod
    def load(cls, path: Path | str | None = None) -> "AutoRAGConfig":
        candidates: list[Path] = []
        if path is not None:
            p = Path(path)
            candidates.append(p)
            # If explicit path missing, still fall through to defaults
        candidates.append(Path("config.yaml"))
        candidates.append(
            Path(platformdirs.user_config_dir("autorag-middleware")) / "config.yaml"
        )
        pkg_root = Path(__file__).resolve().parent.parent
        candidates.append(pkg_root / "configs" / "default.yaml")
        candidates.append(Path("configs") / "default.yaml")

        data: dict[str, Any] = {}
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                with open(candidate, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    data = loaded
                break

        cfg = cls(**data)
        if not cfg.backends:
            cfg.backends["default"] = LLMBackend(
                api_base="http://127.0.0.1:1234/v1",
                api_key="not-needed",
                model="local-model",
            )
        return cfg
