"""
A module that implements tooling to enable easy warnings about deprecations.
"""

import logging
import warnings
from typing import Any, Optional, TextIO, Type, Union

from pip._vendor.packaging.version import parse

from pip import __version__ as current_version  # NOTE: tests patch this name.

DEPRECATION_MSG_PREFIX = "DEPRECATION: "


class PipDeprecationWarning(Warning):
    pass


_original_showwarning: Any = None


# Warnings <-> Logging Integration
def _showwarning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None,
) -> None:
    if file is not None:
        if _original_showwarning is not None:
            _original_showwarning(message, category, filename, lineno, file, line)
    elif issubclass(category, PipDeprecationWarning):
        # We use a specially named logger which will handle all of the
        # deprecation messages for pip.
        logger = logging.getLogger("pip._internal.deprecations")
        logger.warning(message)
    else:
        _original_showwarning(message, category, filename, lineno, file, line)


def install_warning_logger() -> None:
    # Enable our Deprecation Warnings
    warnings.simplefilter("default", PipDeprecationWarning, append=True)

    global _original_showwarning

    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning
        warnings.showwarning = _showwarning


def deprecated(
    *,
    reason: str,
    replacement: Optional[str],
    gone_in: Optional[str],
    feature_flag: Optional[str] = None,
    issue: Optional[int] = None,
) -> None:
    """Helper to deprecate existing functionality.

    reason:
        Textual reason shown to the user about why this functionality has
        been deprecated. Should be a complete sentence.
    replacement:
        Textual suggestion shown to the user about what alternative
        functionality they can use.
    gone_in:
        The version of pip does this functionality should get removed in.
        Raises an error if pip's current version is greater than or equal to
        this.
    feature_flag:
        Command-line flag of the form --use-feature={feature_flag} for testing
        upcoming functionality.
    issue:
        Issue number on the tracker that would serve as a useful place for
        users to find related discussion and provide feedback.
    """

    # Determine whether or not the feature is already gone in this version.
    is_gone = gone_in is not None and parse(current_version) >= parse(gone_in)

    message_parts = [
        (reason, f"{DEPRECATION_MSG_PREFIX}{{}}"),
        (
            gone_in,
            "pip {} will enforce this behaviour change."
            if not is_gone
            else "Since pip {}, this is no longer supported.",
        ),
        (
            replacement,
            "A possible replacement is {}.",
        ),
        (
            feature_flag,
            "You can use the flag --use-feature={} to test the upcoming behaviour."
            if not is_gone
            else None,
        ),
        (
            issue,
            "Discussion can be found at https://github.com/pypa/pip/issues/{}",
        ),
    ]

    message = " ".join(
        format_str.format(value)
        for value, format_str in message_parts
        if format_str is not None and value is not None
    )

    # Raise as an error if this behaviour is deprecated.
    if is_gone:
        raise PipDeprecationWarning(message)

    warnings.warn(message, category=PipDeprecationWarning, stacklevel=2)


class LegacyInstallReason:
    def __init__(
        self,
        reason: str,
        replacement: Optional[str] = None,
        gone_in: Optional[str] = None,
        feature_flag: Optional[str] = None,
        issue: Optional[int] = None,
        emit_after_success: bool = False,
        emit_before_install: bool = False,
    ):
        self._reason = reason
        self._replacement = replacement
        self._gone_in = gone_in
        self._feature_flag = feature_flag
        self._issue = issue
        self.emit_after_success = emit_after_success
        self.emit_before_install = emit_before_install

    def emit_deprecation(self, name: str) -> None:
        deprecated(
            reason=self._reason.format(name=name),
            replacement=self._replacement,
            gone_in=self._gone_in,
            feature_flag=self._feature_flag,
            issue=self._issue,
        )


LegacyInstallReasonFailedBdistWheel = LegacyInstallReason(
    reason=(
        "{name} was installed using the legacy 'setup.py install' "
        "method, because a wheel could not be built for it."
    ),
    replacement="to fix the wheel build issue reported above",
    gone_in="23.1",
    issue=8368,
    emit_after_success=True,
)


LegacyInstallReasonMissingWheelPackage = LegacyInstallReason(
    reason=(
        "{name} is being installed using the legacy "
        "'setup.py install' method, because it does not have a "
        "'pyproject.toml' and the 'wheel' package "
        "is not installed."
    ),
    replacement="to enable the '--use-pep517' option",
    gone_in="23.1",
    issue=8559,
    emit_before_install=True,
)

LegacyInstallReasonNoBinaryForcesSetuptoolsInstall = LegacyInstallReason(
    reason=(
        "{name} is being installed using the legacy "
        "'setup.py install' method, because the '--no-binary' option was enabled "
        "for it and this currently disables local wheel building for projects that "
        "don't have a 'pyproject.toml' file."
    ),
    replacement="to enable the '--use-pep517' option",
    gone_in="23.1",
    issue=11451,
    emit_before_install=True,
)
