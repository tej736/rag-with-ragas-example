from dataclasses import dataclass


SUPPORTED_PROVIDERS = ("openai", "huggingface")


@dataclass
class ProviderConfig:
    provider: str
    model: str

    def validate(self) -> None:
        provider = (self.provider or "").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. Supported providers: {SUPPORTED_PROVIDERS}"
            )
        if not (self.model or "").strip():
            raise ValueError("Model must be provided")


def normalize_provider(provider: str) -> str:
    return (provider or "openai").strip().lower()


def validate_distributions(distributions: dict, tolerance: float = 1e-6) -> None:
    required = {"simple", "reasoning", "multi_context"}
    if set(distributions.keys()) != required:
        raise ValueError(f"Distribution keys must be exactly: {required}")

    for key, value in distributions.items():
        if value < 0:
            raise ValueError(f"Distribution value for '{key}' must be >= 0")

    total = sum(distributions.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Distribution values must sum to 1.0. Got {total}")
