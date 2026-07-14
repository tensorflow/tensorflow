"""Common utilities and argument lists for Python rules wrappers."""

# Attributes unsupported in OSS rules_python and should be stripped.
_UNSUPPORTED_ARGS = [
    "strict_deps",
    "lazy_imports",
    "flaky_test_attempts",
    "linking_mode",
]

def filter_kwargs(kwargs):
    """Filters kwargs to remove unsupported internal attributes."""
    return {k: v for k, v in kwargs.items() if k not in _UNSUPPORTED_ARGS}
