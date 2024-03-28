"""
Utilities for determining application-specific dirs. See <https://github.com/platformdirs/platformdirs> for details and
usage.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 8):  # pragma: no cover (py38+)
    from typing import Literal
else:  # pragma: no cover (py38+)
    from pip._vendor.typing_extensions import Literal

from .api import PlatformDirsABC
from .version import __version__
from .version import __version_tuple__ as __version_info__


def _set_platform_dir_class() -> type[PlatformDirsABC]:
    if sys.platform == "win32":
        from pip._vendor.platformdirs.windows import Windows as Result
    elif sys.platform == "darwin":
        from pip._vendor.platformdirs.macos import MacOS as Result
    else:
        from pip._vendor.platformdirs.unix import Unix as Result

    if os.getenv("ANDROID_DATA") == "/data" and os.getenv("ANDROID_ROOT") == "/system":

        if os.getenv("SHELL") or os.getenv("PREFIX"):
            return Result

        from pip._vendor.platformdirs.android import _android_folder

        if _android_folder() is not None:
            from pip._vendor.platformdirs.android import Android

            return Android  # return to avoid redefinition of result

    return Result


PlatformDirs = _set_platform_dir_class()  #: Currently active platform
AppDirs = PlatformDirs  #: Backwards compatibility with appdirs


def user_data_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    roaming: bool = False,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.version>`.
    :returns: data directory tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, roaming=roaming).user_data_dir


def site_data_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    multipath: bool = False,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `roaming <platformdirs.api.PlatformDirsABC.multipath>`.
    :returns: data directory shared by users
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, multipath=multipath).site_data_dir


def user_config_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    roaming: bool = False,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.version>`.
    :returns: config directory tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, roaming=roaming).user_config_dir


def site_config_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    multipath: bool = False,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `roaming <platformdirs.api.PlatformDirsABC.multipath>`.
    :returns: config directory shared by the users
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, multipath=multipath).site_config_dir


def user_cache_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :returns: cache directory tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, opinion=opinion).user_cache_dir


def user_state_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    roaming: bool = False,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.version>`.
    :returns: state directory tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, roaming=roaming).user_state_dir


def user_log_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :returns: log directory tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, opinion=opinion).user_log_dir


def user_documents_dir() -> str:
    """
    :returns: documents directory tied to the user
    """
    return PlatformDirs().user_documents_dir


def user_runtime_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :returns: runtime directory tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, opinion=opinion).user_runtime_dir


def user_data_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    roaming: bool = False,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.version>`.
    :returns: data path tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, roaming=roaming).user_data_path


def site_data_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    multipath: bool = False,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `multipath <platformdirs.api.PlatformDirsABC.multipath>`.
    :returns: data path shared by users
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, multipath=multipath).site_data_path


def user_config_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    roaming: bool = False,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.version>`.
    :returns: config path tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, roaming=roaming).user_config_path


def site_config_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    multipath: bool = False,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `roaming <platformdirs.api.PlatformDirsABC.multipath>`.
    :returns: config path shared by the users
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, multipath=multipath).site_config_path


def user_cache_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :returns: cache path tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, opinion=opinion).user_cache_path


def user_state_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    roaming: bool = False,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.version>`.
    :returns: state path tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, roaming=roaming).user_state_path


def user_log_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :returns: log path tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, opinion=opinion).user_log_path


def user_documents_path() -> Path:
    """
    :returns: documents path tied to the user
    """
    return PlatformDirs().user_documents_path


def user_runtime_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :returns: runtime path tied to the user
    """
    return PlatformDirs(appname=appname, appauthor=appauthor, version=version, opinion=opinion).user_runtime_path


__all__ = [
    "__version__",
    "__version_info__",
    "PlatformDirs",
    "AppDirs",
    "PlatformDirsABC",
    "user_data_dir",
    "user_config_dir",
    "user_cache_dir",
    "user_state_dir",
    "user_log_dir",
    "user_documents_dir",
    "user_runtime_dir",
    "site_data_dir",
    "site_config_dir",
    "user_data_path",
    "user_config_path",
    "user_cache_path",
    "user_state_path",
    "user_log_path",
    "user_documents_path",
    "user_runtime_path",
    "site_data_path",
    "site_config_path",
]
