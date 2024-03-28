import logging
import os
import re
import string
import typing
from itertools import chain as _chain

_logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# PEP 440

VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_REGEX = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.X | re.I)


def pep440(version: str) -> bool:
    return VERSION_REGEX.match(version) is not None


# -------------------------------------------------------------------------------------
# PEP 508

PEP508_IDENTIFIER_PATTERN = r"([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])"
PEP508_IDENTIFIER_REGEX = re.compile(f"^{PEP508_IDENTIFIER_PATTERN}$", re.I)


def pep508_identifier(name: str) -> bool:
    return PEP508_IDENTIFIER_REGEX.match(name) is not None


try:
    try:
        from packaging import requirements as _req
    except ImportError:  # pragma: no cover
        # let's try setuptools vendored version
        from setuptools._vendor.packaging import requirements as _req  # type: ignore

    def pep508(value: str) -> bool:
        try:
            _req.Requirement(value)
            return True
        except _req.InvalidRequirement:
            return False

except ImportError:  # pragma: no cover
    _logger.warning(
        "Could not find an installation of `packaging`. Requirements, dependencies and "
        "versions might not be validated. "
        "To enforce validation, please install `packaging`."
    )

    def pep508(value: str) -> bool:
        return True


def pep508_versionspec(value: str) -> bool:
    """Expression that can be used to specify/lock versions (including ranges)"""
    if any(c in value for c in (";", "]", "@")):
        # In PEP 508:
        # conditional markers, extras and URL specs are not included in the
        # versionspec
        return False
    # Let's pretend we have a dependency called `requirement` with the given
    # version spec, then we can re-use the pep508 function for validation:
    return pep508(f"requirement{value}")


# -------------------------------------------------------------------------------------
# PEP 517


def pep517_backend_reference(value: str) -> bool:
    module, _, obj = value.partition(":")
    identifiers = (i.strip() for i in _chain(module.split("."), obj.split(".")))
    return all(python_identifier(i) for i in identifiers if i)


# -------------------------------------------------------------------------------------
# Classifiers - PEP 301


def _download_classifiers() -> str:
    import ssl
    from email.message import Message
    from urllib.request import urlopen

    url = "https://pypi.org/pypi?:action=list_classifiers"
    context = ssl.create_default_context()
    with urlopen(url, context=context) as response:
        headers = Message()
        headers["content_type"] = response.getheader("content-type", "text/plain")
        return response.read().decode(headers.get_param("charset", "utf-8"))


class _TroveClassifier:
    """The ``trove_classifiers`` package is the official way of validating classifiers,
    however this package might not be always available.
    As a workaround we can still download a list from PyPI.
    We also don't want to be over strict about it, so simply skipping silently is an
    option (classifiers will be validated anyway during the upload to PyPI).
    """

    def __init__(self):
        self.downloaded: typing.Union[None, False, typing.Set[str]] = None
        self._skip_download = False
        # None => not cached yet
        # False => cache not available
        self.__name__ = "trove_classifier"  # Emulate a public function

    def _disable_download(self):
        # This is a private API. Only setuptools has the consent of using it.
        self._skip_download = True

    def __call__(self, value: str) -> bool:
        if self.downloaded is False or self._skip_download is True:
            return True

        if os.getenv("NO_NETWORK") or os.getenv("VALIDATE_PYPROJECT_NO_NETWORK"):
            self.downloaded = False
            msg = (
                "Install ``trove-classifiers`` to ensure proper validation. "
                "Skipping download of classifiers list from PyPI (NO_NETWORK)."
            )
            _logger.debug(msg)
            return True

        if self.downloaded is None:
            msg = (
                "Install ``trove-classifiers`` to ensure proper validation. "
                "Meanwhile a list of classifiers will be downloaded from PyPI."
            )
            _logger.debug(msg)
            try:
                self.downloaded = set(_download_classifiers().splitlines())
            except Exception:
                self.downloaded = False
                _logger.debug("Problem with download, skipping validation")
                return True

        return value in self.downloaded or value.lower().startswith("private ::")


try:
    from trove_classifiers import classifiers as _trove_classifiers

    def trove_classifier(value: str) -> bool:
        return value in _trove_classifiers or value.lower().startswith("private ::")

except ImportError:  # pragma: no cover
    trove_classifier = _TroveClassifier()


# -------------------------------------------------------------------------------------
# Non-PEP related


def url(value: str) -> bool:
    from urllib.parse import urlparse

    try:
        parts = urlparse(value)
        if not parts.scheme:
            _logger.warning(
                "For maximum compatibility please make sure to include a "
                "`scheme` prefix in your URL (e.g. 'http://'). "
                f"Given value: {value}"
            )
            if not (value.startswith("/") or value.startswith("\\") or "@" in value):
                parts = urlparse(f"http://{value}")

        return bool(parts.scheme and parts.netloc)
    except Exception:
        return False


# https://packaging.python.org/specifications/entry-points/
ENTRYPOINT_PATTERN = r"[^\[\s=]([^=]*[^\s=])?"
ENTRYPOINT_REGEX = re.compile(f"^{ENTRYPOINT_PATTERN}$", re.I)
RECOMMEDED_ENTRYPOINT_PATTERN = r"[\w.-]+"
RECOMMEDED_ENTRYPOINT_REGEX = re.compile(f"^{RECOMMEDED_ENTRYPOINT_PATTERN}$", re.I)
ENTRYPOINT_GROUP_PATTERN = r"\w+(\.\w+)*"
ENTRYPOINT_GROUP_REGEX = re.compile(f"^{ENTRYPOINT_GROUP_PATTERN}$", re.I)


def python_identifier(value: str) -> bool:
    return value.isidentifier()


def python_qualified_identifier(value: str) -> bool:
    if value.startswith(".") or value.endswith("."):
        return False
    return all(python_identifier(m) for m in value.split("."))


def python_module_name(value: str) -> bool:
    return python_qualified_identifier(value)


def python_entrypoint_group(value: str) -> bool:
    return ENTRYPOINT_GROUP_REGEX.match(value) is not None


def python_entrypoint_name(value: str) -> bool:
    if not ENTRYPOINT_REGEX.match(value):
        return False
    if not RECOMMEDED_ENTRYPOINT_REGEX.match(value):
        msg = f"Entry point `{value}` does not follow recommended pattern: "
        msg += RECOMMEDED_ENTRYPOINT_PATTERN
        _logger.warning(msg)
    return True


def python_entrypoint_reference(value: str) -> bool:
    module, _, rest = value.partition(":")
    if "[" in rest:
        obj, _, extras_ = rest.partition("[")
        if extras_.strip()[-1] != "]":
            return False
        extras = (x.strip() for x in extras_.strip(string.whitespace + "[]").split(","))
        if not all(pep508_identifier(e) for e in extras):
            return False
        _logger.warning(f"`{value}` - using extras for entry points is not recommended")
    else:
        obj = rest

    module_parts = module.split(".")
    identifiers = _chain(module_parts, obj.split(".")) if rest else module_parts
    return all(python_identifier(i.strip()) for i in identifiers)
