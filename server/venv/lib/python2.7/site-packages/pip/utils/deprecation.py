"""
A module that implements tooling to enable easy warnings about deprecations.
"""
from __future__ import absolute_import

import logging
import warnings


class PipDeprecationWarning(Warning):
    pass


class Pending(object):
    pass


class RemovedInPip10Warning(PipDeprecationWarning):
    pass


class RemovedInPip11Warning(PipDeprecationWarning, Pending):
    pass


class Python26DeprecationWarning(PipDeprecationWarning):
    pass


# Warnings <-> Logging Integration


_warnings_showwarning = None


def _showwarning(message, category, filename, lineno, file=None, line=None):
    if file is not None:
        if _warnings_showwarning is not None:
            _warnings_showwarning(
                message, category, filename, lineno, file, line,
            )
    else:
        if issubclass(category, PipDeprecationWarning):
            # We use a specially named logger which will handle all of the
            # deprecation messages for pip.
            logger = logging.getLogger("pip.deprecations")

            # This is purposely using the % formatter here instead of letting
            # the logging module handle the interpolation. This is because we
            # want it to appear as if someone typed this entire message out.
            log_message = "DEPRECATION: %s" % message

            # PipDeprecationWarnings that are Pending still have at least 2
            # versions to go until they are removed so they can just be
            # warnings.  Otherwise, they will be removed in the very next
            # version of pip. We want these to be more obvious so we use the
            # ERROR logging level.
            if issubclass(category, Pending):
                logger.warning(log_message)
            else:
                logger.error(log_message)
        else:
            _warnings_showwarning(
                message, category, filename, lineno, file, line,
            )


def install_warning_logger():
    # Enable our Deprecation Warnings
    warnings.simplefilter("default", PipDeprecationWarning, append=True)

    global _warnings_showwarning

    if _warnings_showwarning is None:
        _warnings_showwarning = warnings.showwarning
        warnings.showwarning = _showwarning
