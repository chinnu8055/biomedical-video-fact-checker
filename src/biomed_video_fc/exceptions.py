"""Application-specific exceptions surfaced by pipeline stages."""


class BioFactCheckError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(BioFactCheckError):
    """Raised when required configuration is missing or invalid."""


class MediaProcessingError(BioFactCheckError):
    """Raised when video, audio, or frame extraction fails."""


class ModelUnavailableError(BioFactCheckError):
    """Raised when an optional model dependency is not installed or cannot load."""


class ExternalServiceError(BioFactCheckError):
    """Raised when PubMed or an LLM provider is unavailable."""
