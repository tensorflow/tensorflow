"""A simple log mechanism styled after PEP 282."""

# This module should be kept compatible with Python 2.1.

# The class here is styled after PEP 282 so that it could later be
# replaced with a standard Python logging implementation.

DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4
FATAL = 5

import sys

class Log:

    def __init__(self, threshold=WARN):
        self.threshold = threshold

    def _log(self, level, msg, args):
        if level >= self.threshold:
            if not args:
                # msg may contain a '%'. If args is empty,
                # don't even try to string-format
                print msg
            else:
                print msg % args
            sys.stdout.flush()

    def log(self, level, msg, *args):
        self._log(level, msg, args)

    def debug(self, msg, *args):
        self._log(DEBUG, msg, args)

    def info(self, msg, *args):
        self._log(INFO, msg, args)

    def warn(self, msg, *args):
        self._log(WARN, msg, args)

    def error(self, msg, *args):
        self._log(ERROR, msg, args)

    def fatal(self, msg, *args):
        self._log(FATAL, msg, args)

_global_log = Log()
log = _global_log.log
debug = _global_log.debug
info = _global_log.info
warn = _global_log.warn
error = _global_log.error
fatal = _global_log.fatal

def set_threshold(level):
    # return the old threshold for use from tests
    old = _global_log.threshold
    _global_log.threshold = level
    return old

def set_verbosity(v):
    if v <= 0:
        set_threshold(WARN)
    elif v == 1:
        set_threshold(INFO)
    elif v >= 2:
        set_threshold(DEBUG)
