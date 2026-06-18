import logging

import numpy as np

from model_manager import funasr_profile, normalize_funasr_model_key

log = logging.getLogger("LiveTranslate.FunASR")


class FunASREngine:
    """Unified FunASR engine that dispatches to model-family adapters."""

    def __init__(
        self,
        model_key: str = "sensevoice-small",
        device: str = "cuda",
        hub: str = "ms",
        pad_seconds: float | None = None,
    ):
        self.model_key = normalize_funasr_model_key(model_key)
        self.profile = funasr_profile(self.model_key)
        self.family = self.profile["family"]

        if self.family == "sensevoice":
            from asr_sensevoice import SenseVoiceEngine

            self._engine = SenseVoiceEngine(device=device, hub=hub, pad_seconds=pad_seconds)
        elif self.family == "funasr-nano":
            from asr_funasr_nano import FunASRNanoEngine

            self._engine = FunASRNanoEngine(
                device=device,
                hub=hub,
                engine_type=self.profile["legacy_engine"],
            )
        else:
            raise ValueError(f"Unsupported FunASR model family: {self.family}")

        log.info(
            f"FunASR loaded: {self.profile['display_name']} "
            f"({self.model_key}, family={self.family})"
        )

    def set_language(self, language: str):
        if hasattr(self._engine, "set_language"):
            self._engine.set_language(language)

    def set_input_padding(self, pad_seconds):
        if self.profile.get("supports_padding") and hasattr(
            self._engine, "set_input_padding"
        ):
            self._engine.set_input_padding(pad_seconds)

    def to_device(self, device: str):
        if hasattr(self._engine, "to_device"):
            return self._engine.to_device(device)
        return False

    def unload(self):
        if hasattr(self._engine, "unload"):
            self._engine.unload()

    def transcribe(self, audio: np.ndarray) -> dict | None:
        return self._engine.transcribe(audio)
