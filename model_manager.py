import os
import contextlib
import logging
from pathlib import Path

log = logging.getLogger("LiveTranslate.ModelManager")

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextlib.contextmanager
def _proxy_env(proxy: str):
    """Temporarily route all download backends through a proxy.

    proxy:
        "system" / "" / None -> leave ambient env & OS proxy untouched
        "none"               -> force-disable any proxy for this download
        a URL                -> send urllib/requests/httpx traffic through it

    Covers torch.hub (urllib), huggingface_hub and modelscope (requests),
    which all honor the *_PROXY env vars; urllib additionally gets an explicit
    opener so a previously cached default opener cannot bypass the setting.
    """
    import urllib.request

    if proxy in ("system", "", None):
        yield
        return
    saved_env: dict = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    saved_no_proxy = os.environ.get("NO_PROXY")
    saved_opener = getattr(urllib.request, "_opener", None)
    try:
        if proxy == "none":
            for key in _PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            handler = urllib.request.ProxyHandler({})
        else:
            for key in _PROXY_ENV_KEYS:
                os.environ[key] = proxy
            os.environ.pop("NO_PROXY", None)
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        urllib.request.install_opener(urllib.request.build_opener(handler))
        log.info(f"Download proxy active: {proxy}")
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = saved_no_proxy
        urllib.request.install_opener(saved_opener)

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

ASR_MODEL_IDS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "funasr-mlt-nano": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    "anime-whisper": "litagin/anime-whisper",
}

FUNASR_MODEL_PROFILES = {
    "sensevoice-small": {
        "display_name": "SenseVoice Small",
        "family": "sensevoice",
        "legacy_engine": "sensevoice",
        "modelscope_id": "iic/SenseVoiceSmall",
        "huggingface_id": "FunAudioLLM/SenseVoiceSmall",
        "estimated_bytes": 940_000_000,
        "supports_padding": True,
        "supports_language": True,
    },
    "funasr-nano-2512": {
        "display_name": "Fun-ASR-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "estimated_bytes": 1_050_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
    "funasr-mlt-nano-2512": {
        "display_name": "Fun-ASR-MLT-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-mlt-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "estimated_bytes": 1_050_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
}

DEFAULT_FUNASR_MODEL = "sensevoice-small"

FUNASR_LEGACY_ENGINE_ALIASES = {
    "sensevoice": "sensevoice-small",
    "funasr-nano": "funasr-nano-2512",
    "funasr-mlt-nano": "funasr-mlt-nano-2512",
}

# HuggingFace repo ids for engines whose namespace differs from ModelScope.
# SenseVoice lives under `iic/` on ModelScope but `FunAudioLLM/` on HuggingFace.
ASR_MODEL_IDS_HF = {
    "sensevoice": "FunAudioLLM/SenseVoiceSmall",
}


def asr_model_id(
    engine_type: str, hub: str = "ms", funasr_model: str | None = None
) -> str:
    """Return the repo id for an engine on the given hub ('ms' or 'hf')."""
    if engine_type == "funasr":
        return funasr_model_id(funasr_model, hub)
    if engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        return funasr_model_id(FUNASR_LEGACY_ENGINE_ALIASES[engine_type], hub)
    if hub == "hf" and engine_type in ASR_MODEL_IDS_HF:
        return ASR_MODEL_IDS_HF[engine_type]
    return ASR_MODEL_IDS[engine_type]

ASR_DISPLAY_NAMES = {
    "funasr": "FunASR",
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
    "funasr-mlt-nano": "Fun-ASR-MLT-Nano",
    "whisper": "Whisper",
    "anime-whisper": "Anime-Whisper",
    "crispasr": "CrispASR",
}

CRISPASR_MODEL_PROFILES = {
    "parakeet-tdt-0.6b-v3-q4_k": {
        "display_name": "Parakeet TDT 0.6B v3 Q4_K",
        "repo_id": "cstr/parakeet-tdt-0.6b-v3-GGUF",
        "filename": "parakeet-tdt-0.6b-v3-q4_k.gguf",
        "backend": "auto",
        "estimated_bytes": 650_000_000,
        "languages": "25 EU",
        "output_style": "native_punctuation",
        "needs_punctuation": False,
        "sha256": None,
    },
}

DEFAULT_CRISPASR_MODEL = "parakeet-tdt-0.6b-v3-q4_k"
_CRISPASR_EXTS = {".gguf", ".bin"}
_CRISPASR_MIN_BYTES = 1_000_000

_MODEL_SIZE_BYTES = {
    "silero-vad": 2_000_000,
    "sensevoice": 940_000_000,
    "funasr-nano": 1_050_000_000,
    "funasr-mlt-nano": 1_050_000_000,
    "whisper-tiny": 78_000_000,
    "whisper-base": 148_000_000,
    "whisper-small": 488_000_000,
    "whisper-medium": 1_530_000_000,
    "whisper-large-v3": 3_100_000_000,
    "anime-whisper": 3_100_000_000,
    "crispasr:parakeet-tdt-0.6b-v3-q4_k": 650_000_000,
}

_WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

_CACHE_MODELS = [
    ("SenseVoice Small", "funasr", "sensevoice-small"),
    ("Fun-ASR-Nano", "funasr", "funasr-nano-2512"),
    ("Fun-ASR-MLT-Nano", "funasr", "funasr-mlt-nano-2512"),
    ("Anime-Whisper", "anime-whisper"),
]


def normalize_funasr_model_key(model_key: str | None) -> str:
    if model_key in FUNASR_MODEL_PROFILES:
        return model_key
    if model_key in FUNASR_LEGACY_ENGINE_ALIASES:
        return FUNASR_LEGACY_ENGINE_ALIASES[model_key]
    return DEFAULT_FUNASR_MODEL


def normalize_asr_engine_selection(
    engine_type: str | None, funasr_model: str | None = None
) -> tuple[str, str]:
    if engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        return "funasr", FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
    if engine_type == "funasr":
        return "funasr", normalize_funasr_model_key(funasr_model)
    return engine_type or "funasr", normalize_funasr_model_key(funasr_model)


def migrate_funasr_settings(settings: dict | None) -> dict | None:
    if not settings:
        return settings
    engine, model_key = normalize_asr_engine_selection(
        settings.get("asr_engine"), settings.get("funasr_model")
    )
    settings["asr_engine"] = engine
    if engine == "funasr":
        settings["funasr_model"] = model_key
    else:
        settings.setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    return settings


def funasr_profile(model_key: str | None) -> dict:
    return FUNASR_MODEL_PROFILES[normalize_funasr_model_key(model_key)]


def funasr_model_options() -> list[tuple[str, str]]:
    return [
        (key, profile["display_name"])
        for key, profile in FUNASR_MODEL_PROFILES.items()
    ]


def funasr_display_name(model_key: str | None) -> str:
    return funasr_profile(model_key)["display_name"]


def funasr_supports_padding(model_key: str | None) -> bool:
    return bool(funasr_profile(model_key).get("supports_padding"))


def funasr_model_id(model_key: str | None, hub: str = "ms") -> str:
    profile = funasr_profile(model_key)
    return profile["huggingface_id"] if hub == "hf" else profile["modelscope_id"]


def normalize_crispasr_model_key(model_key: str | None) -> str:
    if model_key in CRISPASR_MODEL_PROFILES:
        return model_key
    return DEFAULT_CRISPASR_MODEL


def crispasr_profile(model_key: str | None) -> dict:
    return CRISPASR_MODEL_PROFILES[normalize_crispasr_model_key(model_key)]


def crispasr_model_options() -> list[tuple[str, str]]:
    return [
        (key, profile["display_name"])
        for key, profile in CRISPASR_MODEL_PROFILES.items()
    ]


def is_crispasr_model_file(path) -> bool:
    if not path:
        return False
    path = Path(path)
    try:
        return (
            path.is_file()
            and path.suffix.lower() in _CRISPASR_EXTS
            and path.stat().st_size >= _CRISPASR_MIN_BYTES
        )
    except OSError:
        return False


def _custom_crispasr_path(value) -> Path | None:
    if not value or value in CRISPASR_MODEL_PROFILES:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def resolve_custom_crispasr_model(value) -> str | None:
    path = _custom_crispasr_path(value)
    if path and is_crispasr_model_file(path):
        return str(path.absolute())
    return None


def _hf_repo_cache_dir(repo_id: str) -> Path:
    org, name = repo_id.split("/", 1)
    return MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}"


def _hf_cached_file(repo_id: str, filename: str) -> Path | None:
    snap_root = _hf_repo_cache_dir(repo_id) / "snapshots"
    if not snap_root.exists():
        return None
    try:
        snapshots = sorted(
            (p for p in snap_root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except (OSError, PermissionError):
        return None
    for snap in snapshots:
        path = snap / filename
        if is_crispasr_model_file(path):
            return path
    return None


def _crispasr_legacy_model_path(model_key: str | None) -> Path:
    profile = crispasr_profile(model_key)
    return MODELS_DIR / "crispasr" / profile["filename"]


def crispasr_model_path(model_key: str | None, hub: str = "hf") -> str:
    profile = crispasr_profile(model_key)
    cached = _hf_cached_file(profile["repo_id"], profile["filename"])
    if cached:
        return str(cached.resolve())
    legacy = _crispasr_legacy_model_path(model_key)
    return str(legacy.resolve())


def _custom_whisper_path(value) -> Path | None:
    if not value or value in _WHISPER_SIZES:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def is_faster_whisper_model_dir(path) -> bool:
    """True when path looks like a CTranslate2 faster-whisper model directory."""
    if not path:
        return False
    path = Path(path)
    return (
        path.is_dir()
        and (path / "model.bin").is_file()
        and (path / "config.json").is_file()
    )


def resolve_custom_whisper_model(value) -> str | None:
    path = _custom_whisper_path(value)
    if path and is_faster_whisper_model_dir(path):
        return str(path.resolve())
    return None


def _is_builtin_whisper_cache(path: Path) -> bool:
    parts = set(path.parts)
    return any(f"models--Systran--faster-whisper-{s}" in parts for s in _WHISPER_SIZES)


def _is_builtin_crispasr_cache(path: Path) -> bool:
    parts = set(path.parts)
    for profile in CRISPASR_MODEL_PROFILES.values():
        org, name = profile["repo_id"].split("/", 1)
        if f"models--{org}--{name}" in parts:
            return True
    return False


def _is_hf_hub_cache(path: Path) -> bool:
    parts = path.parts
    marker = ("huggingface", "hub")
    return any(parts[i : i + 2] == marker for i in range(len(parts) - 1))


def _hf_snapshot_name(path: Path) -> str | None:
    """Return 'org/repo' for .../models--org--repo/snapshots/<hash>."""
    if path.parent.name != "snapshots":
        return None
    repo_dir = path.parent.parent
    if not repo_dir.name.startswith("models--"):
        return None

    encoded = repo_dir.name.removeprefix("models--")
    parts = encoded.split("--", 1)
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def list_local_faster_whisper_models() -> list[dict]:
    """Scan ./models for user-provided faster-whisper model directories."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        model_bins = list(MODELS_DIR.rglob("model.bin"))
    except (OSError, PermissionError):
        return []

    for model_bin in model_bins:
        model_dir = model_bin.parent
        if _is_builtin_whisper_cache(model_dir):
            continue
        if not is_faster_whisper_model_dir(model_dir):
            continue
        try:
            resolved = str(model_dir.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        name = _hf_snapshot_name(model_dir) or model_dir.name
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{name} ({model_dir.name[:8]})"
        entries.append({"name": name, "path": resolved})

    entries.sort(key=lambda item: item["name"].lower())
    return entries

def local_faster_whisper_display_name(path) -> str | None:
    """Return the same display name used by the local Whisper model selector."""
    resolved = resolve_custom_whisper_model(path)
    if not resolved:
        return None
    for item in list_local_faster_whisper_models():
        if item["path"] == resolved:
            return item["name"]
    return _hf_snapshot_name(Path(resolved)) or Path(resolved).name


def list_local_crispasr_models() -> list[dict]:
    """Scan ./models for user-provided CrispASR single-file models."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        files = [
            path
            for ext in _CRISPASR_EXTS
            for path in MODELS_DIR.rglob(f"*{ext}")
        ]
    except (OSError, PermissionError):
        return []

    for path in files:
        if _is_builtin_crispasr_cache(path):
            continue
        if path.name == "model.bin" and is_faster_whisper_model_dir(path.parent):
            continue
        if not is_crispasr_model_file(path):
            continue
        try:
            identity = str(path.resolve())
            model_path = str(path.absolute())
        except OSError:
            continue
        if identity in seen:
            continue
        seen.add(identity)

        name = path.name
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{path.stem} ({path.parent.name}){path.suffix}"
        entries.append({"name": name, "path": model_path})

    entries.sort(key=lambda item: item["name"].lower())
    return entries


def local_crispasr_display_name(path) -> str | None:
    resolved = resolve_custom_crispasr_model(path)
    if not resolved:
        return None
    for item in list_local_crispasr_models():
        if item["path"] == resolved:
            return item["name"]
    return Path(resolved).name


def apply_cache_env():
    """Point all model caches to ./models/."""
    resolved = str(MODELS_DIR.resolve())
    os.environ["MODELSCOPE_CACHE"] = os.path.join(resolved, "modelscope")
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(resolved, "torch")
    log.info(f"Cache env set: {resolved}")


def _has_silero_pkg() -> bool:
    """True when the silero-vad PyPI package (model bundled in wheel) is installed."""
    import importlib.util

    return importlib.util.find_spec("silero_vad") is not None


def is_silero_cached() -> bool:
    if _has_silero_pkg():
        return True
    torch_hub = MODELS_DIR / "torch" / "hub"
    return any(torch_hub.glob("snakers4_silero-vad*")) if torch_hub.exists() else False


def _ms_model_path(org, name):
    """Return the first existing ModelScope cache path, or the default."""
    for sub in (
        MODELS_DIR / "modelscope" / org / name,
        MODELS_DIR / "modelscope" / "hub" / "models" / org / name,
    ):
        if sub.exists():
            return sub
    return MODELS_DIR / "modelscope" / org / name


def _hf_repo_complete(org: str, name: str, min_bytes: int = 50_000_000) -> bool:
    """True if a HuggingFace repo cache exists AND finished downloading.

    A killed/aborted download leaves snapshot entries pointing at missing blobs
    (broken symlinks) or '.incomplete' blobs; treating that as cached makes the
    model load hang. Validate a snapshot where every file resolves (stat follows
    symlinks; a broken link raises) and the resolved bytes are substantial. This
    ignores orphan '.incomplete' blobs left behind by an earlier interrupted run.
    """
    snap_root = MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    if not snap_root.exists():
        return False
    for snap in snap_root.iterdir():
        if not snap.is_dir():
            continue
        total = 0
        broken = False
        for f in snap.rglob("*"):
            if f.is_dir():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                broken = True
                break
        if not broken and total >= min_bytes:
            return True
    return False


def is_asr_cached(engine_type, model_size="medium", hub="ms") -> bool:
    if engine_type == "crispasr":
        custom = resolve_custom_crispasr_model(model_size)
        if custom:
            return True
        if model_size not in CRISPASR_MODEL_PROFILES:
            return False
        profile = crispasr_profile(model_size)
        if _hf_cached_file(profile["repo_id"], profile["filename"]):
            return True
        return is_crispasr_model_file(_crispasr_legacy_model_path(model_size))
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(model_size)
        )
        # Accept cache from either hub to avoid redundant downloads; the repo
        # namespace can differ between ModelScope and HuggingFace (SenseVoice).
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        if _ms_model_path(ms_org, ms_name).exists():
            return True
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
        if _hf_repo_complete(hf_org, hf_name):
            return True
        return False
    if engine_type == "anime-whisper":
        # HF-only (not published to ModelScope). Check that snapshots dir actually
        # contains weight files; an .incomplete blob means a prior run aborted mid-download.
        model_id = ASR_MODEL_IDS[engine_type]
        org, name = model_id.split("/")
        snap_root = (
            MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        )
        if not snap_root.exists():
            return False
        for snap in snap_root.iterdir():
            if not snap.is_dir():
                continue
            has_weights = any(
                (snap / fn).exists()
                for fn in ("model.safetensors", "pytorch_model.bin")
            )
            has_config = (snap / "config.json").exists()
            if has_weights and has_config:
                return True
        return False
    elif engine_type == "whisper":
        if model_size not in _WHISPER_SIZES:
            return resolve_custom_whisper_model(model_size) is not None
        min_bytes = int(
            _MODEL_SIZE_BYTES.get(f"whisper-{model_size}", 50_000_000) * 0.5
        )
        return _hf_repo_complete(
            "Systran", f"faster-whisper-{model_size}", min_bytes=min_bytes
        )
    return True


def get_missing_models(engine, model_size, hub) -> list:
    missing = []
    if not is_silero_cached():
        missing.append(
            {
                "name": "Silero VAD",
                "type": "silero-vad",
                "estimated_bytes": _MODEL_SIZE_BYTES["silero-vad"],
            }
        )
    if not is_asr_cached(engine, model_size, hub):
        if engine == "whisper" and model_size not in _WHISPER_SIZES:
            return missing
        if engine == "crispasr" and model_size not in CRISPASR_MODEL_PROFILES:
            return missing
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            profile = funasr_profile(model_key)
            key = f"funasr:{model_key}"
            display = profile["display_name"]
            estimated_bytes = profile["estimated_bytes"]
        elif engine == "whisper":
            key = engine if engine != "whisper" else f"whisper-{model_size}"
            display = f"Whisper {model_size}"
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        elif engine == "crispasr":
            profile = crispasr_profile(model_size)
            key = f"crispasr:{normalize_crispasr_model_key(model_size)}"
            display = f"CrispASR {profile['display_name']}"
            estimated_bytes = profile["estimated_bytes"]
        else:
            key = engine
            display = ASR_DISPLAY_NAMES.get(engine, engine)
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        missing.append(
            {
                "name": display,
                "type": key,
                "estimated_bytes": estimated_bytes,
            }
        )
    return missing


def get_local_model_path(engine_type, hub="ms", funasr_model: str | None = None):
    """Return local snapshot path if model is cached, else None.

    Checks the preferred hub first, then falls back to the other hub.
    """
    if engine_type == "crispasr":
        model_value = funasr_model
        custom = resolve_custom_crispasr_model(model_value)
        if custom:
            return custom
        if model_value not in CRISPASR_MODEL_PROFILES:
            return None
        profile = crispasr_profile(model_value)
        path = _hf_cached_file(profile["repo_id"], profile["filename"])
        if path:
            return str(path.resolve())
        legacy = _crispasr_legacy_model_path(model_value)
        return str(legacy.resolve()) if is_crispasr_model_file(legacy) else None

    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(funasr_model)
        )
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
    elif engine_type in ASR_MODEL_IDS:
        ms_org, ms_name = asr_model_id(engine_type, "ms").split("/")
        hf_org, hf_name = asr_model_id(engine_type, "hf").split("/")
    else:
        return None

    def _try_ms():
        local = _ms_model_path(ms_org, ms_name)
        return str(local) if local.exists() else None

    def _try_hf():
        snap_dir = (
            MODELS_DIR
            / "huggingface"
            / "hub"
            / f"models--{hf_org}--{hf_name}"
            / "snapshots"
        )
        if snap_dir.exists():
            snaps = sorted(snap_dir.iterdir())
            if snaps:
                return str(snaps[-1])
        return None

    if hub == "ms":
        return _try_ms() or _try_hf()
    else:
        return _try_hf() or _try_ms()


def download_silero(proxy: str = "system"):
    if _has_silero_pkg():
        log.info("Silero VAD bundled by silero-vad package, no download needed")
        return
    import torch

    log.info("Downloading Silero VAD...")
    with _proxy_env(proxy):
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:master",
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as exc:
            if "CERTIFICATE_VERIFY" not in str(exc):
                raise
            log.warning("SSL strict verification failed, retrying with relaxed flags")
            model, _ = _load_silero_relaxed_ssl()
    del model
    log.info("Silero VAD downloaded")


def _load_silero_relaxed_ssl():
    # Python 3.13 enables VERIFY_X509_STRICT by default, rejecting certificates
    # without an Authority Key Identifier (common behind SSL-inspecting proxies).
    import ssl

    import torch

    strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    original = ssl._create_default_https_context

    def relaxed_context(*args, **kwargs):
        ctx = ssl.create_default_context(*args, **kwargs)
        ctx.verify_flags &= ~strict
        return ctx

    ssl._create_default_https_context = relaxed_context
    try:
        return torch.hub.load(
            repo_or_dir="snakers4/silero-vad:master",
            model="silero_vad",
            trust_repo=True,
            force_reload=True,
        )
    finally:
        ssl._create_default_https_context = original


def download_asr(engine, model_size="medium", hub="ms", proxy="system"):
    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")
    with _proxy_env(proxy):
        if engine == "crispasr":
            if model_size not in CRISPASR_MODEL_PROFILES:
                raise ValueError(f"Invalid CrispASR model: {model_size}")
            from huggingface_hub import hf_hub_download

            profile = crispasr_profile(model_size)
            log.info(
                f"Downloading {profile['repo_id']}:{profile['filename']} "
                "from HuggingFace..."
            )
            hf_hub_download(
                repo_id=profile["repo_id"],
                filename=profile["filename"],
                cache_dir=hf_cache,
            )
        elif engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            if hub == "ms":
                from modelscope import snapshot_download

                model_id = funasr_model_id(model_key, "ms")
                log.info(f"Downloading {model_id} from ModelScope...")
                snapshot_download(model_id=model_id, cache_dir=ms_cache)
            else:
                from huggingface_hub import snapshot_download

                model_id = funasr_model_id(model_key, "hf")
                log.info(f"Downloading {model_id} from HuggingFace...")
                snapshot_download(repo_id=model_id, cache_dir=hf_cache)
        elif engine == "anime-whisper":
            # HF-only, ignore hub setting
            from huggingface_hub import snapshot_download

            model_id = ASR_MODEL_IDS[engine]
            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
        elif engine == "whisper":
            if model_size not in _WHISPER_SIZES:
                raise ValueError(f"Invalid local faster-whisper model: {model_size}")
            from huggingface_hub import snapshot_download

            model_id = f"Systran/faster-whisper-{model_size}"
            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
    log.info(f"ASR model downloaded: {engine}")


def dir_size(path) -> int:
    path = Path(path)
    if path.is_file():
        try:
            return path.stat().st_size
        except (OSError, PermissionError):
            return 0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def get_cache_entries():
    """Scan ./models/ for cached models."""
    entries = []
    hf_base = MODELS_DIR / "huggingface" / "hub"
    torch_base = MODELS_DIR / "torch" / "hub"

    for entry in _CACHE_MODELS:
        if len(entry) == 3:
            name, engine, model_key = entry
        else:
            name, engine = entry
            model_key = None
        ms_org, ms_model = asr_model_id(engine, "ms", model_key).split("/")
        hf_org, hf_model = asr_model_id(engine, "hf", model_key).split("/")
        ms_path = _ms_model_path(ms_org, ms_model)
        hf_path = hf_base / f"models--{hf_org}--{hf_model}"
        if ms_path.exists():
            entries.append((f"{name} (ModelScope)", ms_path))
        if hf_path.exists():
            entries.append((f"{name} (HuggingFace)", hf_path))

    for profile in CRISPASR_MODEL_PROFILES.values():
        repo_dir = _hf_repo_cache_dir(profile["repo_id"])
        if _hf_cached_file(profile["repo_id"], profile["filename"]):
            entries.append((f"CrispASR {profile['display_name']} (HuggingFace)", repo_dir))

    for size in _WHISPER_SIZES:
        hf_path = hf_base / f"models--Systran--faster-whisper-{size}"
        if hf_path.exists() and is_asr_cached("whisper", size, "hf"):
            entries.append((f"Whisper {size}", hf_path))

    for item in list_local_faster_whisper_models():
        entries.append((f"Whisper Local: {item['name']}", Path(item["path"])))

    for item in list_local_crispasr_models():
        entries.append((f"CrispASR Local: {item['name']}", Path(item["path"])))

    if torch_base.exists():
        for d in sorted(torch_base.glob("snakers4_silero-vad*")):
            if d.is_dir():
                entries.append(("Silero VAD", d))
                break

    return entries
