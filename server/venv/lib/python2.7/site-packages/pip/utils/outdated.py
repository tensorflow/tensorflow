from __future__ import absolute_import

import datetime
import json
import logging
import os.path
import sys

from pip._vendor import lockfile
from pip._vendor.packaging import version as packaging_version

from pip.compat import total_seconds, WINDOWS
from pip.models import PyPI
from pip.locations import USER_CACHE_DIR, running_under_virtualenv
from pip.utils import ensure_dir, get_installed_version
from pip.utils.filesystem import check_path_owner


SELFCHECK_DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"


logger = logging.getLogger(__name__)


class VirtualenvSelfCheckState(object):
    def __init__(self):
        self.statefile_path = os.path.join(sys.prefix, "pip-selfcheck.json")

        # Load the existing state
        try:
            with open(self.statefile_path) as statefile:
                self.state = json.load(statefile)
        except (IOError, ValueError):
            self.state = {}

    def save(self, pypi_version, current_time):
        # Attempt to write out our version check file
        with open(self.statefile_path, "w") as statefile:
            json.dump(
                {
                    "last_check": current_time.strftime(SELFCHECK_DATE_FMT),
                    "pypi_version": pypi_version,
                },
                statefile,
                sort_keys=True,
                separators=(",", ":")
            )


class GlobalSelfCheckState(object):
    def __init__(self):
        self.statefile_path = os.path.join(USER_CACHE_DIR, "selfcheck.json")

        # Load the existing state
        try:
            with open(self.statefile_path) as statefile:
                self.state = json.load(statefile)[sys.prefix]
        except (IOError, ValueError, KeyError):
            self.state = {}

    def save(self, pypi_version, current_time):
        # Check to make sure that we own the directory
        if not check_path_owner(os.path.dirname(self.statefile_path)):
            return

        # Now that we've ensured the directory is owned by this user, we'll go
        # ahead and make sure that all our directories are created.
        ensure_dir(os.path.dirname(self.statefile_path))

        # Attempt to write out our version check file
        with lockfile.LockFile(self.statefile_path):
            if os.path.exists(self.statefile_path):
                with open(self.statefile_path) as statefile:
                    state = json.load(statefile)
            else:
                state = {}

            state[sys.prefix] = {
                "last_check": current_time.strftime(SELFCHECK_DATE_FMT),
                "pypi_version": pypi_version,
            }

            with open(self.statefile_path, "w") as statefile:
                json.dump(state, statefile, sort_keys=True,
                          separators=(",", ":"))


def load_selfcheck_statefile():
    if running_under_virtualenv():
        return VirtualenvSelfCheckState()
    else:
        return GlobalSelfCheckState()


def pip_version_check(session):
    """Check for an update for pip.

    Limit the frequency of checks to once per week. State is stored either in
    the active virtualenv or in the user's USER_CACHE_DIR keyed off the prefix
    of the pip script path.
    """
    installed_version = get_installed_version("pip")
    if installed_version is None:
        return

    pip_version = packaging_version.parse(installed_version)
    pypi_version = None

    try:
        state = load_selfcheck_statefile()

        current_time = datetime.datetime.utcnow()
        # Determine if we need to refresh the state
        if "last_check" in state.state and "pypi_version" in state.state:
            last_check = datetime.datetime.strptime(
                state.state["last_check"],
                SELFCHECK_DATE_FMT
            )
            if total_seconds(current_time - last_check) < 7 * 24 * 60 * 60:
                pypi_version = state.state["pypi_version"]

        # Refresh the version if we need to or just see if we need to warn
        if pypi_version is None:
            resp = session.get(
                PyPI.pip_json_url,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            pypi_version = [
                v for v in sorted(
                    list(resp.json()["releases"]),
                    key=packaging_version.parse,
                )
                if not packaging_version.parse(v).is_prerelease
            ][-1]

            # save that we've performed a check
            state.save(pypi_version, current_time)

        remote_version = packaging_version.parse(pypi_version)

        # Determine if our pypi_version is older
        if (pip_version < remote_version and
                pip_version.base_version != remote_version.base_version):
            # Advise "python -m pip" on Windows to avoid issues
            # with overwriting pip.exe.
            if WINDOWS:
                pip_cmd = "python -m pip"
            else:
                pip_cmd = "pip"
            logger.warning(
                "You are using pip version %s, however version %s is "
                "available.\nYou should consider upgrading via the "
                "'%s install --upgrade pip' command.",
                pip_version, pypi_version, pip_cmd
            )

    except Exception:
        logger.debug(
            "There was an error checking the latest version of pip",
            exc_info=True,
        )
