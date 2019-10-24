"""
    This module provides mechanisms to use signal handlers in Python.

    Functions:

    signal(sig,action) -- set the action for a given signal (done)
    pause(sig) -- wait until a signal arrives [Unix only]
    alarm(seconds) -- cause SIGALRM after a specified time [Unix only]
    getsignal(sig) -- get the signal action for a given signal
    default_int_handler(action) -- default SIGINT handler (done, but acts string)

    Constants:

    SIG_DFL -- used to refer to the system default handler
    SIG_IGN -- used to ignore the signal
    NSIG -- number of defined signals

    SIGINT, SIGTERM, etc. -- signal numbers

    *** IMPORTANT NOTICES ***
    A signal handler function is called with two arguments:
    the first is the signal number, the second is the interrupted stack frame.

    According to http://java.sun.com/products/jdk/faq/faq-sun-packages.html
    'writing java programs that rely on sun.* is risky: they are not portable, and are not supported.'

    However, in Jython, like Python, we let you decide what makes
    sense for your application. If sun.misc.Signal is not available,
    an ImportError is raised.
"""


try:
    import sun.misc.Signal
except ImportError:
    raise ImportError("signal module requires sun.misc.Signal, which is not available on this platform")

import os
import sun.misc.SignalHandler
import sys
import threading
import time
from java.lang import IllegalArgumentException
from java.util.concurrent.atomic import AtomicReference

debug = 0

def _init_signals():
    # install signals by checking for standard names
    # using IllegalArgumentException to diagnose

    possible_signals = """
        SIGABRT
        SIGALRM
        SIGBUS
        SIGCHLD
        SIGCONT
        SIGFPE
        SIGHUP
        SIGILL
        SIGINFO
        SIGINT
        SIGIOT
        SIGKILL
        SIGPIPE
        SIGPOLL
        SIGPROF
        SIGQUIT
        SIGSEGV
        SIGSTOP
        SIGSYS
        SIGTERM
        SIGTRAP
        SIGTSTP
        SIGTTIN
        SIGTTOU
        SIGURG
        SIGUSR1
        SIGUSR2
        SIGVTALRM
        SIGWINCH
        SIGXCPU
        SIGXFSZ
    """.split()

    _module = __import__(__name__)
    signals = {}
    signals_by_name = {}
    for signal_name in possible_signals:
        try:
            java_signal = sun.misc.Signal(signal_name[3:])
        except IllegalArgumentException:
            continue

        signal_number = java_signal.getNumber()
        signals[signal_number] = java_signal
        signals_by_name[signal_name] = java_signal
        setattr(_module, signal_name, signal_number) # install as a module constant
    return signals

_signals = _init_signals()
NSIG = max(_signals.iterkeys()) + 1
SIG_DFL = sun.misc.SignalHandler.SIG_DFL # default system handler
SIG_IGN = sun.misc.SignalHandler.SIG_IGN # handler to ignore a signal

class JythonSignalHandler(sun.misc.SignalHandler):
    def __init__(self, action):
        self.action = action

    def handle(self, signal):
        # passing a frame here probably don't make sense in a threaded system,
        # but perhaps revisit
        self.action(signal.getNumber(), None)

def signal(sig, action):
    """
    signal(sig, action) -> action

    Set the action for the given signal.  The action can be SIG_DFL,
    SIG_IGN, or a callable Python object.  The previous action is
    returned.  See getsignal() for possible return values.

    *** IMPORTANT NOTICE ***
    A signal handler function is called with two arguments:
    the first is the signal number, the second is the interrupted stack frame.
    """
    # maybe keep a weak ref map of handlers we have returned?

    try:
        signal = _signals[sig]
    except KeyError:
        raise ValueError("signal number out of range")

    if callable(action):
        prev = sun.misc.Signal.handle(signal, JythonSignalHandler(action))
    elif action in (SIG_IGN, SIG_DFL) or isinstance(action, sun.misc.SignalHandler):
        prev = sun.misc.Signal.handle(signal, action)
    else:
        raise TypeError("signal handler must be signal.SIG_IGN, signal.SIG_DFL, or a callable object")

    if isinstance(prev, JythonSignalHandler):
        return prev.action
    else:
        return prev


# dangerous! don't use!
def getsignal(sig):
    """getsignal(sig) -> action

    Return the current action for the given signal.  The return value can be:
    SIG_IGN -- if the signal is being ignored
    SIG_DFL -- if the default action for the signal is in effect
    None -- if an unknown handler is in effect
    anything else -- the callable Python object used as a handler

    Note for Jython: this function is NOT threadsafe. The underlying
    Java support only enables getting the current signal handler by
    setting a new one. So this is completely prone to race conditions.
    """
    try:
        signal = _signals[sig]
    except KeyError:
        raise ValueError("signal number out of range")
    current = sun.misc.Signal.handle(signal, SIG_DFL)
    sun.misc.Signal.handle(signal, current) # and reinstall

    if isinstance(current, JythonSignalHandler):
        return current.action
    else:
        return current

def default_int_handler(sig, frame):
    """
    default_int_handler(...)

    The default handler for SIGINT installed by Python.
    It raises KeyboardInterrupt.
    """
    raise KeyboardInterrupt

def pause():
    raise NotImplementedError

_alarm_timer_holder = AtomicReference()

def _alarm_handler(sig, frame):
    print "Alarm clock"
    os._exit(0)

# install a default alarm handler, the one we get by default doesn't
# work terribly well since it throws a bus error (at least on OS X)!
try:
    SIGALRM
    signal(SIGALRM, _alarm_handler)
except NameError:
    pass

class _Alarm(object):
    def __init__(self, interval, task):
        self.interval = interval
        self.task = task
        self.scheduled = None
        self.timer = threading.Timer(self.interval, self.task)

    def start(self):
        self.timer.start()
        self.scheduled = time.time() + self.interval

    def cancel(self):
        self.timer.cancel()
        now = time.time()
        if self.scheduled and self.scheduled > now:
            return self.scheduled - now
        else:
            return 0

def alarm(time):
    try:
        SIGALRM
    except NameError:
        raise NotImplementedError("alarm not implemented on this platform")

    def raise_alarm():
        sun.misc.Signal.raise(_signals[SIGALRM])

    if time > 0:
        new_alarm_timer = _Alarm(time, raise_alarm)
    else:
        new_alarm_timer = None
    old_alarm_timer = _alarm_timer_holder.getAndSet(new_alarm_timer)
    if old_alarm_timer:
        scheduled = int(old_alarm_timer.cancel())
    else:
        scheduled = 0

    if new_alarm_timer:
        new_alarm_timer.start()
    return scheduled
