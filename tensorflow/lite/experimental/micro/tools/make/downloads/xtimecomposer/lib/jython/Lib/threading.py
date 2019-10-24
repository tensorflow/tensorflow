from java.lang import InterruptedException
from java.util import Collections, WeakHashMap
from java.util.concurrent import Semaphore, CyclicBarrier
from java.util.concurrent.locks import ReentrantLock
from org.python.util import jython
from thread import _newFunctionThread
from thread import _local as local
from _threading import Lock, RLock, Condition, _Lock, _RLock
import java.lang.Thread
import weakref

import sys as _sys
from traceback import print_exc as _print_exc

# Rename some stuff so "from threading import *" is safe
__all__ = ['activeCount', 'Condition', 'currentThread', 'enumerate', 'Event',
           'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Thread',
           'Timer', 'setprofile', 'settrace', 'local', 'stack_size']

_VERBOSE = False

if __debug__:

    class _Verbose(object):

        def __init__(self, verbose=None):
            if verbose is None:
                verbose = _VERBOSE
            self.__verbose = verbose

        def _note(self, format, *args):
            if self.__verbose:
                format = format % args
                format = "%s: %s\n" % (
                    currentThread().getName(), format)
                _sys.stderr.write(format)

else:
    # Disable this when using "python -O"
    class _Verbose(object):
        def __init__(self, verbose=None):
            pass
        def _note(self, *args):
            pass

# Support for profile and trace hooks

_profile_hook = None
_trace_hook = None

def setprofile(func):
    global _profile_hook
    _profile_hook = func

def settrace(func):
    global _trace_hook
    _trace_hook = func


class Semaphore(object):
    def __init__(self, value=1):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._semaphore = java.util.concurrent.Semaphore(value)

    def acquire(self, blocking=True):
        if blocking:
            self._semaphore.acquire()
            return True
        else:
            return self._semaphore.tryAcquire()

    def __enter__(self):
        self.acquire()
        return self

    def release(self):
        self._semaphore.release()

    def __exit__(self, t, v, tb):
        self.release()


ThreadStates = {
    java.lang.Thread.State.NEW : 'initial',
    java.lang.Thread.State.RUNNABLE: 'started',
    java.lang.Thread.State.BLOCKED: 'started',
    java.lang.Thread.State.WAITING: 'started',
    java.lang.Thread.State.TIMED_WAITING: 'started',
    java.lang.Thread.State.TERMINATED: 'stopped',
}

class JavaThread(object):
    def __init__(self, thread):
        self._thread = thread
        _jthread_to_pythread[thread] = self
        _threads[thread.getId()] = self

    def __repr__(self):
        _thread = self._thread
        status = ThreadStates[_thread.getState()]
        if _thread.isDaemon(): status + " daemon"
        return "<%s(%s, %s)>" % (self.__class__.__name__, self.getName(), status)

    def __eq__(self, other):
        if isinstance(other, JavaThread):
            return self._thread == other._thread
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def start(self):
        self._thread.start()

    def run(self):
        self._thread.run()

    def join(self, timeout=None):
        if timeout:
            millis = timeout * 1000.
            millis_int = int(millis)
            nanos = int((millis - millis_int) * 1e6)
            self._thread.join(millis_int, nanos)
        else:
            self._thread.join()

    def getName(self):
        return self._thread.getName()

    def setName(self, name):
        self._thread.setName(str(name))

    def isAlive(self):
        return self._thread.isAlive()

    def isDaemon(self):
        return self._thread.isDaemon()

    def setDaemon(self, daemonic):
        self._thread.setDaemon(bool(daemonic))

# relies on the fact that this is a CHM
_threads = weakref.WeakValueDictionary()
_active = _threads
_jthread_to_pythread = Collections.synchronizedMap(WeakHashMap())

class Thread(JavaThread):
    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None):
        assert group is None, "group argument must be None for now"
        _thread = self._create_thread()
        JavaThread.__init__(self, _thread)
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        self._target = target
        self._args = args
        self._kwargs = kwargs
        if name:
            self._thread.setName(str(name))

    def _create_thread(self):
        return _newFunctionThread(self.__bootstrap, ())

    def run(self):
        if self._target:
            self._target(*self._args, **self._kwargs)

    def __bootstrap(self):
        try:
            if _trace_hook:
                _sys.settrace(_trace_hook)
            if _profile_hook:
                _sys.setprofile(_profile_hook)
            try:
                self.run()
            except SystemExit:
                pass
            except InterruptedException:
                # Quiet InterruptedExceptions if they're caused by
                # _systemrestart
                if not jython.shouldRestart:
                    raise
            except:
                # If sys.stderr is no more (most likely from interpreter
                # shutdown) use self.__stderr.  Otherwise still use sys (as in
                # _sys) in case sys.stderr was redefined.
                if _sys:
                    _sys.stderr.write("Exception in thread %s:" %
                            self.getName())
                    _print_exc(file=_sys.stderr)
                else:
                    # Do the best job possible w/o a huge amt. of code to
                    # approx. a traceback stack trace
                    exc_type, exc_value, exc_tb = self.__exc_info()
                    try:
                        print>>self.__stderr, (
                            "Exception in thread " + self.getName() +
                            " (most likely raised during interpreter shutdown):")
                        print>>self.__stderr, (
                            "Traceback (most recent call last):")
                        while exc_tb:
                            print>>self.__stderr, (
                                '  File "%s", line %s, in %s' %
                                (exc_tb.tb_frame.f_code.co_filename,
                                    exc_tb.tb_lineno,
                                    exc_tb.tb_frame.f_code.co_name))
                            exc_tb = exc_tb.tb_next
                        print>>self.__stderr, ("%s: %s" % (exc_type, exc_value))
                    # Make sure that exc_tb gets deleted since it is a memory
                    # hog; deleting everything else is just for thoroughness
                    finally:
                        del exc_type, exc_value, exc_tb

        finally:
            self.__stop()
            try:
                self.__delete()
            except:
                pass

    def __stop(self):
        pass

    def __delete(self):
        del _threads[self._thread.getId()]


class _MainThread(Thread):
    def __init__(self):
        Thread.__init__(self, name="MainThread")
        import atexit
        atexit.register(self.__exitfunc)

    def _create_thread(self):
        return java.lang.Thread.currentThread()

    def _set_daemon(self):
        return False

    def __exitfunc(self):
        del _threads[self._thread.getId()]
        t = _pickSomeNonDaemonThread()
        while t:
            t.join()
            t = _pickSomeNonDaemonThread()

def _pickSomeNonDaemonThread():
    for t in enumerate():
        if not t.isDaemon() and t.isAlive():
            return t
    return None

def currentThread():
    jthread = java.lang.Thread.currentThread()
    pythread = _jthread_to_pythread[jthread]
    if pythread is None:
        pythread = JavaThread(jthread)
    return pythread

def activeCount():
    return len(_threads)

def enumerate():
    return _threads.values()

from thread import stack_size


_MainThread()


######################################################################
# pure Python code from CPythonLib/threading.py

# The timer class was contributed by Itamar Shtull-Trauring

def Timer(*args, **kwargs):
    return _Timer(*args, **kwargs)

class _Timer(Thread):
    """Call a function after a specified number of seconds:

    t = Timer(30.0, f, args=[], kwargs={})
    t.start()
    t.cancel() # stop the timer's action if it's still waiting
    """

    def __init__(self, interval, function, args=[], kwargs={}):
        Thread.__init__(self)
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet"""
        self.finished.set()

    def run(self):
        self.finished.wait(self.interval)
        if not self.finished.isSet():
            self.function(*self.args, **self.kwargs)
        self.finished.set()


# NOT USED except by BoundedSemaphore
class _Semaphore(_Verbose):

    # After Tim Peters' semaphore class, but not quite the same (no maximum)

    def __init__(self, value=1, verbose=None):
        assert value >= 0, "Semaphore initial value must be >= 0"
        _Verbose.__init__(self, verbose)
        self.__cond = Condition(Lock())
        self.__value = value

    def acquire(self, blocking=1):
        rc = False
        self.__cond.acquire()
        while self.__value == 0:
            if not blocking:
                break
            if __debug__:
                self._note("%s.acquire(%s): blocked waiting, value=%s",
                           self, blocking, self.__value)
            self.__cond.wait()
        else:
            self.__value = self.__value - 1
            if __debug__:
                self._note("%s.acquire: success, value=%s",
                           self, self.__value)
            rc = True
        self.__cond.release()
        return rc

    def release(self):
        self.__cond.acquire()
        self.__value = self.__value + 1
        if __debug__:
            self._note("%s.release: success, value=%s",
                       self, self.__value)
        self.__cond.notify()
        self.__cond.release()


def BoundedSemaphore(*args, **kwargs):
    return _BoundedSemaphore(*args, **kwargs)

class _BoundedSemaphore(_Semaphore):
    """Semaphore that checks that # releases is <= # acquires"""
    def __init__(self, value=1, verbose=None):
        _Semaphore.__init__(self, value, verbose)
        self._initial_value = value

    def __enter__(self):
        self.acquire()
        return self

    def release(self):
        if self._Semaphore__value >= self._initial_value:
            raise ValueError, "Semaphore released too many times"
        return _Semaphore.release(self)

    def __exit__(self, t, v, tb):
        self.release()


def Event(*args, **kwargs):
    return _Event(*args, **kwargs)

class _Event(_Verbose):

    # After Tim Peters' event class (without is_posted())

    def __init__(self, verbose=None):
        _Verbose.__init__(self, verbose)
        self.__cond = Condition(Lock())
        self.__flag = False

    def isSet(self):
        return self.__flag

    def set(self):
        self.__cond.acquire()
        try:
            self.__flag = True
            self.__cond.notifyAll()
        finally:
            self.__cond.release()

    def clear(self):
        self.__cond.acquire()
        try:
            self.__flag = False
        finally:
            self.__cond.release()

    def wait(self, timeout=None):
        self.__cond.acquire()
        try:
            if not self.__flag:
                self.__cond.wait(timeout)
        finally:
            self.__cond.release()
