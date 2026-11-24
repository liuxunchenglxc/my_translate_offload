class StaleAttempt(RuntimeError):
    """Raised when a newer attempt has taken ownership of the task."""
