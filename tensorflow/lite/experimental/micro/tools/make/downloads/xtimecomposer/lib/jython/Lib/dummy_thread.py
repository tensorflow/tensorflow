"""Drop-in replacement for the thread module.

Meant to be used as a brain-dead substitute so that threaded code does
not need to be rewritten for when the thread module is not present.

Suggested usage is::

    try:
        import thread
    except ImportError:
        import dummy_thread as thread

"""
__author__ = "Brett Cannon"
__email__ = "brett@python.org"

# Exports only things specified by thread documentation
# (skipping obsolete synonyms allocate(), start_new(), exit_thread())
__all__ = ['error', 'start_new_thread', 'exit', 'get_ident', 'allocate_lock',
           'interrupt_main', 'LockType']

import traceback as _traceback
import warnings

class error(Exception):
    """Dummy implementation of thread.error."""

    def __init__(self, *args):
        self.args = args

def start_new_thread(function, args, kwargs={}):
    """Dummy implementation of thread.start_new_thread().

    Compatibility is maintained by making sure that ``args`` is a
    tuple and ``kwargs`` is a dictionary.  If an exception is raised
    and it is SystemExit (which can be done by thread.exit()) it is
    caught and nothing is done; all other exceptions are printed out
    by using traceback.print_exc().

    If the executed function calls interrupt_main the KeyboardInterrupt will be
    raised when the function returns.

    """
    if type(args) != type(tuple()):
        raise TypeError("2nd arg must be a tuple")
    if type(kwargs) != type(dict()):
        raise TypeError("3rd arg must be a dict")
    global _main
    _main = False
    try:
        function(*args, **kwargs)
    except SystemExit:
        pass
    except:
        _traceback.print_exc()
    _main = True
    global _interrupt
    if _interrupt:
        _interrupt = False
        raise KeyboardInterrupt

def exit():
    """Dummy implementation of thread.exit()."""
    raise SystemExit

def get_ident():
    """Dummy implementation of thread.get_ident().

    Since this module should only be used when threadmodule is not
    available, it is safe to assume that the current process is the
    only thread.  Thus a constant can be safely returned.
    """
    return -1

def allocate_lock():
    """Dummy implementation of thread.allocate_lock()."""
    return LockType()

def stack_size(size=None):
    """Dummy implementation of thread.stack_size()."""
    if size is not None:
        raise error("setting thread stack size not supported")
    return 0

class LockType(object):
    """Class implementing dummy implementation of thread.LockType.

    Compatibility is maintained by maintaining self.locked_status
    which is a boolean that stores the state of the lock.  Pickling of
    the lock, though, should not be done since if the thread module is
    then used with an unpickled ``lock()`` from here problems could
    occur from this class not having atomic methods.

    """

    def __init__(self):
        self.locked_status = False

    def acquire(self, waitflag=None):
        """Dummy implementation of acquire().

        For blocking calls, self.locked_status is automatically set to
        True and returned appropriately based on value of
        ``waitflag``.  If it is non-blocking, then the value is
        actually checked and not set if it is already acquired.  This
        is all done so that threading.Condition's assert statements
        aren't triggered and throw a little fit.

        """
        if waitflag is None or waitflag:
            self.locked_status = True
            return True
        else:
            if not self.locked_status:
                self.locked_status = True
                return True
            else:
                return False

    __enter__ = acquire

    def __exit__(self, typ, val, tb):
        self.release()

    def release(self):
        """Release the dummy lock."""
        # XXX Perhaps shouldn't actually bother to test?  Could lead
        #     to problems for complex, threaded code.
        if not self.locked_status:
            raise error
        self.locked_status = False
        return True

    def locked(self):
        return self.locked_status

# Used to signal that interrupt_main was called in a "thread"
_interrupt = False
# True when not executing in a "thread"
_main = True

def interrupt_main():
    """Set _interrupt flag to True to have start_new_thread raise
    KeyboardInterrupt upon exiting."""
    if _main:
        raise KeyboardInterrupt
    else:
        global _interrupt
        _interrupt = True
