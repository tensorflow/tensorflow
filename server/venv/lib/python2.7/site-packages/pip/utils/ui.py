from __future__ import absolute_import
from __future__ import division

import itertools
import sys
from signal import signal, SIGINT, default_int_handler
import time
import contextlib
import logging

from pip.compat import WINDOWS
from pip.utils import format_size
from pip.utils.logging import get_indentation
from pip._vendor import six
from pip._vendor.progress.bar import Bar, IncrementalBar
from pip._vendor.progress.helpers import (WritelnMixin,
                                          HIDE_CURSOR, SHOW_CURSOR)
from pip._vendor.progress.spinner import Spinner

try:
    from pip._vendor import colorama
# Lots of different errors can come from this, including SystemError and
# ImportError.
except Exception:
    colorama = None

logger = logging.getLogger(__name__)


def _select_progress_class(preferred, fallback):
    encoding = getattr(preferred.file, "encoding", None)

    # If we don't know what encoding this file is in, then we'll just assume
    # that it doesn't support unicode and use the ASCII bar.
    if not encoding:
        return fallback

    # Collect all of the possible characters we want to use with the preferred
    # bar.
    characters = [
        getattr(preferred, "empty_fill", six.text_type()),
        getattr(preferred, "fill", six.text_type()),
    ]
    characters += list(getattr(preferred, "phases", []))

    # Try to decode the characters we're using for the bar using the encoding
    # of the given file, if this works then we'll assume that we can use the
    # fancier bar and if not we'll fall back to the plaintext bar.
    try:
        six.text_type().join(characters).encode(encoding)
    except UnicodeEncodeError:
        return fallback
    else:
        return preferred


_BaseBar = _select_progress_class(IncrementalBar, Bar)


class InterruptibleMixin(object):
    """
    Helper to ensure that self.finish() gets called on keyboard interrupt.

    This allows downloads to be interrupted without leaving temporary state
    (like hidden cursors) behind.

    This class is similar to the progress library's existing SigIntMixin
    helper, but as of version 1.2, that helper has the following problems:

    1. It calls sys.exit().
    2. It discards the existing SIGINT handler completely.
    3. It leaves its own handler in place even after an uninterrupted finish,
       which will have unexpected delayed effects if the user triggers an
       unrelated keyboard interrupt some time after a progress-displaying
       download has already completed, for example.
    """

    def __init__(self, *args, **kwargs):
        """
        Save the original SIGINT handler for later.
        """
        super(InterruptibleMixin, self).__init__(*args, **kwargs)

        self.original_handler = signal(SIGINT, self.handle_sigint)

        # If signal() returns None, the previous handler was not installed from
        # Python, and we cannot restore it. This probably should not happen,
        # but if it does, we must restore something sensible instead, at least.
        # The least bad option should be Python's default SIGINT handler, which
        # just raises KeyboardInterrupt.
        if self.original_handler is None:
            self.original_handler = default_int_handler

    def finish(self):
        """
        Restore the original SIGINT handler after finishing.

        This should happen regardless of whether the progress display finishes
        normally, or gets interrupted.
        """
        super(InterruptibleMixin, self).finish()
        signal(SIGINT, self.original_handler)

    def handle_sigint(self, signum, frame):
        """
        Call self.finish() before delegating to the original SIGINT handler.

        This handler should only be in place while the progress display is
        active.
        """
        self.finish()
        self.original_handler(signum, frame)


class DownloadProgressMixin(object):

    def __init__(self, *args, **kwargs):
        super(DownloadProgressMixin, self).__init__(*args, **kwargs)
        self.message = (" " * (get_indentation() + 2)) + self.message

    @property
    def downloaded(self):
        return format_size(self.index)

    @property
    def download_speed(self):
        # Avoid zero division errors...
        if self.avg == 0.0:
            return "..."
        return format_size(1 / self.avg) + "/s"

    @property
    def pretty_eta(self):
        if self.eta:
            return "eta %s" % self.eta_td
        return ""

    def iter(self, it, n=1):
        for x in it:
            yield x
            self.next(n)
        self.finish()


class WindowsMixin(object):

    def __init__(self, *args, **kwargs):
        # The Windows terminal does not support the hide/show cursor ANSI codes
        # even with colorama. So we'll ensure that hide_cursor is False on
        # Windows.
        # This call neds to go before the super() call, so that hide_cursor
        # is set in time. The base progress bar class writes the "hide cursor"
        # code to the terminal in its init, so if we don't set this soon
        # enough, we get a "hide" with no corresponding "show"...
        if WINDOWS and self.hide_cursor:
            self.hide_cursor = False

        super(WindowsMixin, self).__init__(*args, **kwargs)

        # Check if we are running on Windows and we have the colorama module,
        # if we do then wrap our file with it.
        if WINDOWS and colorama:
            self.file = colorama.AnsiToWin32(self.file)
            # The progress code expects to be able to call self.file.isatty()
            # but the colorama.AnsiToWin32() object doesn't have that, so we'll
            # add it.
            self.file.isatty = lambda: self.file.wrapped.isatty()
            # The progress code expects to be able to call self.file.flush()
            # but the colorama.AnsiToWin32() object doesn't have that, so we'll
            # add it.
            self.file.flush = lambda: self.file.wrapped.flush()


class DownloadProgressBar(WindowsMixin, InterruptibleMixin,
                          DownloadProgressMixin, _BaseBar):

    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"


class DownloadProgressSpinner(WindowsMixin, InterruptibleMixin,
                              DownloadProgressMixin, WritelnMixin, Spinner):

    file = sys.stdout
    suffix = "%(downloaded)s %(download_speed)s"

    def next_phase(self):
        if not hasattr(self, "_phaser"):
            self._phaser = itertools.cycle(self.phases)
        return next(self._phaser)

    def update(self):
        message = self.message % self
        phase = self.next_phase()
        suffix = self.suffix % self
        line = ''.join([
            message,
            " " if message else "",
            phase,
            " " if suffix else "",
            suffix,
        ])

        self.writeln(line)


################################################################
# Generic "something is happening" spinners
#
# We don't even try using progress.spinner.Spinner here because it's actually
# simpler to reimplement from scratch than to coerce their code into doing
# what we need.
################################################################

@contextlib.contextmanager
def hidden_cursor(file):
    # The Windows terminal does not support the hide/show cursor ANSI codes,
    # even via colorama. So don't even try.
    if WINDOWS:
        yield
    # We don't want to clutter the output with control characters if we're
    # writing to a file, or if the user is running with --quiet.
    # See https://github.com/pypa/pip/issues/3418
    elif not file.isatty() or logger.getEffectiveLevel() > logging.INFO:
        yield
    else:
        file.write(HIDE_CURSOR)
        try:
            yield
        finally:
            file.write(SHOW_CURSOR)


class RateLimiter(object):
    def __init__(self, min_update_interval_seconds):
        self._min_update_interval_seconds = min_update_interval_seconds
        self._last_update = 0

    def ready(self):
        now = time.time()
        delta = now - self._last_update
        return delta >= self._min_update_interval_seconds

    def reset(self):
        self._last_update = time.time()


class InteractiveSpinner(object):
    def __init__(self, message, file=None, spin_chars="-\\|/",
                 # Empirically, 8 updates/second looks nice
                 min_update_interval_seconds=0.125):
        self._message = message
        if file is None:
            file = sys.stdout
        self._file = file
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._finished = False

        self._spin_cycle = itertools.cycle(spin_chars)

        self._file.write(" " * get_indentation() + self._message + " ... ")
        self._width = 0

    def _write(self, status):
        assert not self._finished
        # Erase what we wrote before by backspacing to the beginning, writing
        # spaces to overwrite the old text, and then backspacing again
        backup = "\b" * self._width
        self._file.write(backup + " " * self._width + backup)
        # Now we have a blank slate to add our status
        self._file.write(status)
        self._width = len(status)
        self._file.flush()
        self._rate_limiter.reset()

    def spin(self):
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._write(next(self._spin_cycle))

    def finish(self, final_status):
        if self._finished:
            return
        self._write(final_status)
        self._file.write("\n")
        self._file.flush()
        self._finished = True


# Used for dumb terminals, non-interactive installs (no tty), etc.
# We still print updates occasionally (once every 60 seconds by default) to
# act as a keep-alive for systems like Travis-CI that take lack-of-output as
# an indication that a task has frozen.
class NonInteractiveSpinner(object):
    def __init__(self, message, min_update_interval_seconds=60):
        self._message = message
        self._finished = False
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._update("started")

    def _update(self, status):
        assert not self._finished
        self._rate_limiter.reset()
        logger.info("%s: %s", self._message, status)

    def spin(self):
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._update("still running...")

    def finish(self, final_status):
        if self._finished:
            return
        self._update("finished with status '%s'" % (final_status,))
        self._finished = True


@contextlib.contextmanager
def open_spinner(message):
    # Interactive spinner goes directly to sys.stdout rather than being routed
    # through the logging system, but it acts like it has level INFO,
    # i.e. it's only displayed if we're at level INFO or better.
    # Non-interactive spinner goes through the logging system, so it is always
    # in sync with logging configuration.
    if sys.stdout.isatty() and logger.getEffectiveLevel() <= logging.INFO:
        spinner = InteractiveSpinner(message)
    else:
        spinner = NonInteractiveSpinner(message)
    try:
        with hidden_cursor(sys.stdout):
            yield spinner
    except KeyboardInterrupt:
        spinner.finish("canceled")
        raise
    except Exception:
        spinner.finish("error")
        raise
    else:
        spinner.finish("done")
