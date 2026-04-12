"""Domain exception hierarchy."""


class ModelManagerError(Exception):
    """Base exception for all model_manager errors."""


# ── Hardware ──────────────────────────────────────────────────────────────────

class HardwareDetectionError(ModelManagerError):
    """Failed to detect hardware information."""


# ── Catalog ───────────────────────────────────────────────────────────────────

class CatalogError(ModelManagerError):
    """Base for catalog errors."""

class ModelNotFoundError(CatalogError):
    """Requested model ID not found in catalog."""

class CatalogLoadError(CatalogError):
    """Failed to load or parse catalog file."""


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(ModelManagerError):
    """Base for storage errors."""

class InsufficientDiskSpaceError(StorageError):
    """Not enough free disk space for the requested operation."""
    def __init__(self, required_gb: float, available_gb: float, path: str):
        self.required_gb  = required_gb
        self.available_gb = available_gb
        self.path         = path
        super().__init__(
            f"Need {required_gb:.1f} GB but only {available_gb:.1f} GB free at {path}"
        )


# ── Permissions ───────────────────────────────────────────────────────────────

class PermissionError(ModelManagerError):
    """Base for permission errors."""

class ElevationDeniedError(PermissionError):
    """User denied privilege escalation request."""

class ElevationFailedError(PermissionError):
    """Platform-level elevation attempt failed."""


# ── Backends ──────────────────────────────────────────────────────────────────

class BackendError(ModelManagerError):
    """Base for installation backend errors."""

class BackendNotAvailableError(BackendError):
    """Requested installation backend is not available in this environment."""

class InstallError(BackendError):
    """A package installation step failed."""
    def __init__(self, message: str, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)


# ── Repository / download ─────────────────────────────────────────────────────

class RepositoryError(ModelManagerError):
    """Base for repository/download errors."""

class DownloadError(RepositoryError):
    """File download failed."""

class ChecksumMismatchError(RepositoryError):
    """Downloaded file checksum does not match expected value."""


# ── Agent / LLM ───────────────────────────────────────────────────────────────

class AgentError(ModelManagerError):
    """Base for AI agent errors."""

class APIKeyMissingError(AgentError):
    """Anthropic API key is not configured."""

class DiagnosisFailedError(AgentError):
    """Error diagnosis agent could not produce a fix plan."""


# ── State / session ───────────────────────────────────────────────────────────

class StateError(ModelManagerError):
    """Base for state machine / session errors."""

class SessionNotFoundError(StateError):
    """Requested session ID does not exist."""

class InvalidTransitionError(StateError):
    """Attempted state transition is not allowed."""


# ── Recovery ──────────────────────────────────────────────────────────────────

class RecoveryError(ModelManagerError):
    """Base for branch-and-resume errors."""

class BranchDepthExceededError(RecoveryError):
    """Error recovery branch nesting exceeded the maximum allowed depth."""

class BranchFailedError(RecoveryError):
    """All fix plans in the recovery branch were exhausted without success."""
