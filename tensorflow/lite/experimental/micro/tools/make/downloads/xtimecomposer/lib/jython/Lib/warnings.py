"""Python part of the warnings subsystem."""

# Note: function level imports should *not* be used
# in this module as it may cause import lock deadlock.
# See bug 683658.
import sys, types
import linecache

__all__ = ["warn", "showwarning", "formatwarning", "filterwarnings",
           "resetwarnings"]

# filters contains a sequence of filter 5-tuples
# The components of the 5-tuple are:
# - an action: error, ignore, always, default, module, or once
# - a compiled regex that must match the warning message
# - a class representing the warning category
# - a compiled regex that must match the module that is being warned
# - a line number for the line being warning, or 0 to mean any line
# If either if the compiled regexs are None, match anything.
filters = []
defaultaction = "default"
onceregistry = {}

def warn(message, category=None, stacklevel=1):
    """Issue a warning, or maybe ignore it or raise an exception."""
    # Check if message is already a Warning object
    if isinstance(message, Warning):
        category = message.__class__
    # Check category argument
    if category is None:
        category = UserWarning
    assert issubclass(category, Warning)
    # Get context information
    try:
        caller = sys._getframe(stacklevel)
    except ValueError:
        globals = sys.__dict__
        lineno = 1
    else:
        globals = caller.f_globals
        lineno = caller.f_lineno
    if '__name__' in globals:
        module = globals['__name__']
    else:
        module = "<string>"
    filename = globals.get('__file__')
    if filename:
        fnl = filename.lower()
        if fnl.endswith((".pyc", ".pyo")):
            filename = filename[:-1]
        elif fnl.endswith("$py.class"):
            filename = filename[:-9] + '.py'
    else:
        if module == "__main__":
            try:
                filename = sys.argv[0]
            except AttributeError:
                # embedded interpreters don't have sys.argv, see bug #839151
                filename = '__main__'
        if not filename:
            filename = module
    registry = globals.setdefault("__warningregistry__", {})
    warn_explicit(message, category, filename, lineno, module, registry,
                  globals)

def warn_explicit(message, category, filename, lineno,
                  module=None, registry=None, module_globals=None):
    if module is None:
        module = filename or "<unknown>"
        if module[-3:].lower() == ".py":
            module = module[:-3] # XXX What about leading pathname?
    if registry is None:
        registry = {}
    if isinstance(message, Warning):
        text = str(message)
        category = message.__class__
    else:
        text = message
        message = category(message)
    key = (text, category, lineno)
    # Quick test for common case
    if registry.get(key):
        return
    # Search the filters
    for item in filters:
        action, msg, cat, mod, ln = item
        if ((msg is None or msg.match(text)) and
            issubclass(category, cat) and
            (mod is None or mod.match(module)) and
            (ln == 0 or lineno == ln)):
            break
    else:
        action = defaultaction
    # Early exit actions
    if action == "ignore":
        registry[key] = 1
        return

    # Prime the linecache for formatting, in case the
    # "file" is actually in a zipfile or something.
    linecache.getlines(filename, module_globals)

    if action == "error":
        raise message
    # Other actions
    if action == "once":
        registry[key] = 1
        oncekey = (text, category)
        if onceregistry.get(oncekey):
            return
        onceregistry[oncekey] = 1
    elif action == "always":
        pass
    elif action == "module":
        registry[key] = 1
        altkey = (text, category, 0)
        if registry.get(altkey):
            return
        registry[altkey] = 1
    elif action == "default":
        registry[key] = 1
    else:
        # Unrecognized actions are errors
        raise RuntimeError(
              "Unrecognized action (%r) in warnings.filters:\n %s" %
              (action, item))
    # Print message and context
    showwarning(message, category, filename, lineno)

def showwarning(message, category, filename, lineno, file=None):
    """Hook to write a warning to a file; replace if you like."""
    if file is None:
        file = sys.stderr
    try:
        file.write(formatwarning(message, category, filename, lineno))
    except IOError:
        pass # the file (probably stderr) is invalid - this warning gets lost.

def formatwarning(message, category, filename, lineno):
    """Function to format a warning the standard way."""
    s =  "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)
    line = linecache.getline(filename, lineno).strip()
    if line:
        s = s + "  " + line + "\n"
    return s

def filterwarnings(action, message="", category=Warning, module="", lineno=0,
                   append=0):
    """Insert an entry into the list of warnings filters (at the front).

    Use assertions to check that all arguments have the right type."""
    import re
    assert action in ("error", "ignore", "always", "default", "module",
                      "once"), "invalid action: %r" % (action,)
    assert isinstance(message, basestring), "message must be a string"
    assert isinstance(category, (type, types.ClassType)), \
           "category must be a class"
    assert issubclass(category, Warning), "category must be a Warning subclass"
    assert isinstance(module, basestring), "module must be a string"
    assert isinstance(lineno, int) and lineno >= 0, \
           "lineno must be an int >= 0"
    item = (action, re.compile(message, re.I), category,
            re.compile(module), lineno)
    if append:
        filters.append(item)
    else:
        filters.insert(0, item)

def simplefilter(action, category=Warning, lineno=0, append=0):
    """Insert a simple entry into the list of warnings filters (at the front).

    A simple filter matches all modules and messages.
    """
    assert action in ("error", "ignore", "always", "default", "module",
                      "once"), "invalid action: %r" % (action,)
    assert isinstance(lineno, int) and lineno >= 0, \
           "lineno must be an int >= 0"
    item = (action, None, category, None, lineno)
    if append:
        filters.append(item)
    else:
        filters.insert(0, item)

def resetwarnings():
    """Clear the list of warning filters, so that no filters are active."""
    filters[:] = []

class _OptionError(Exception):
    """Exception used by option processing helpers."""
    pass

# Helper to process -W options passed via sys.warnoptions
def _processoptions(args):
    for arg in args:
        try:
            _setoption(arg)
        except _OptionError, msg:
            print >>sys.stderr, "Invalid -W option ignored:", msg

# Helper for _processoptions()
def _setoption(arg):
    import re
    parts = arg.split(':')
    if len(parts) > 5:
        raise _OptionError("too many fields (max 5): %r" % (arg,))
    while len(parts) < 5:
        parts.append('')
    action, message, category, module, lineno = [s.strip()
                                                 for s in parts]
    action = _getaction(action)
    message = re.escape(message)
    category = _getcategory(category)
    module = re.escape(module)
    if module:
        module = module + '$'
    if lineno:
        try:
            lineno = int(lineno)
            if lineno < 0:
                raise ValueError
        except (ValueError, OverflowError):
            raise _OptionError("invalid lineno %r" % (lineno,))
    else:
        lineno = 0
    filterwarnings(action, message, category, module, lineno)

# Helper for _setoption()
def _getaction(action):
    if not action:
        return "default"
    if action == "all": return "always" # Alias
    for a in ('default', 'always', 'ignore', 'module', 'once', 'error'):
        if a.startswith(action):
            return a
    raise _OptionError("invalid action: %r" % (action,))

# Helper for _setoption()
def _getcategory(category):
    import re
    if not category:
        return Warning
    if re.match("^[a-zA-Z0-9_]+$", category):
        try:
            cat = eval(category)
        except NameError:
            raise _OptionError("unknown warning category: %r" % (category,))
    else:
        i = category.rfind(".")
        module = category[:i]
        klass = category[i+1:]
        try:
            m = __import__(module, None, None, [klass])
        except ImportError:
            raise _OptionError("invalid module name: %r" % (module,))
        try:
            cat = getattr(m, klass)
        except AttributeError:
            raise _OptionError("unknown warning category: %r" % (category,))
    if not issubclass(cat, Warning):
        raise _OptionError("invalid warning category: %r" % (category,))
    return cat

# Module initialization
_processoptions(sys.warnoptions)
simplefilter("ignore", category=PendingDeprecationWarning, append=1)
simplefilter("ignore", category=ImportWarning, append=1)
