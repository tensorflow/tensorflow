"""Common pathname manipulations, JDK version.

Instead of importing this module directly, import os and refer to this
module as os.path.

"""

# Incompletely implemented:
# ismount -- How?
# normcase -- How?

# Missing:
# sameopenfile -- Java doesn't have fstat nor file descriptors?
# samestat -- How?

import stat
import sys
from java.io import File
import java.io.IOException
from java.lang import System
import os

from org.python.core.Py import newString as asPyString


def _tostr(s, method):
    if isinstance(s, basestring):
        return s
    raise TypeError, "%s() argument must be a str or unicode object, not %s" % (
                method, _type_name(s))

def _type_name(obj):
    TPFLAGS_HEAPTYPE = 1 << 9
    type_name = ''
    obj_type = type(obj)
    is_heap = obj_type.__flags__ & TPFLAGS_HEAPTYPE == TPFLAGS_HEAPTYPE
    if not is_heap and obj_type.__module__ != '__builtin__':
        type_name = '%s.' % obj_type.__module__
    type_name += obj_type.__name__
    return type_name

def dirname(path):
    """Return the directory component of a pathname"""
    path = _tostr(path, "dirname")
    result = asPyString(File(path).getParent())
    if not result:
        if isabs(path):
            result = path # Must be root
        else:
            result = ""
    return result

def basename(path):
    """Return the final component of a pathname"""
    path = _tostr(path, "basename")
    return asPyString(File(path).getName())

def split(path):
    """Split a pathname.

    Return tuple "(head, tail)" where "tail" is everything after the
    final slash.  Either part may be empty.

    """
    path = _tostr(path, "split")
    return (dirname(path), basename(path))

def splitext(path):
    """Split the extension from a pathname.

    Extension is everything from the last dot to the end.  Return
    "(root, ext)", either part may be empty.

    """
    i = 0
    n = -1
    for c in path:
        if c == '.': n = i
        i = i+1
    if n < 0:
        return (path, "")
    else:
        return (path[:n], path[n:])

def splitdrive(path):
    """Split a pathname into drive and path specifiers.

    Returns a 2-tuple "(drive,path)"; either part may be empty.
    """
    # Algorithm based on CPython's ntpath.splitdrive and ntpath.isabs.
    if path[1:2] == ':' and path[0].lower() in 'abcdefghijklmnopqrstuvwxyz' \
            and (path[2:] == '' or path[2] in '/\\'):
        return path[:2], path[2:]
    return '', path

def exists(path):
    """Test whether a path exists.

    Returns false for broken symbolic links.

    """
    path = _tostr(path, "exists")
    return File(sys.getPath(path)).exists()

def isabs(path):
    """Test whether a path is absolute"""
    path = _tostr(path, "isabs")
    return File(path).isAbsolute()

def isfile(path):
    """Test whether a path is a regular file"""
    path = _tostr(path, "isfile")
    return File(sys.getPath(path)).isFile()

def isdir(path):
    """Test whether a path is a directory"""
    path = _tostr(path, "isdir")
    return File(sys.getPath(path)).isDirectory()

def join(path, *args):
    """Join two or more pathname components, inserting os.sep as needed"""
    path = _tostr(path, "join")
    f = File(path)
    for a in args:
        a = _tostr(a, "join")
        g = File(a)
        if g.isAbsolute() or len(f.getPath()) == 0:
            f = g
        else:
            if a == "":
                a = os.sep
            f = File(f, a)
    return asPyString(f.getPath())

def normcase(path):
    """Normalize case of pathname.

    XXX Not done right under JDK.

    """
    path = _tostr(path, "normcase")
    return asPyString(File(path).getPath())

def commonprefix(m):
    "Given a list of pathnames, return the longest common leading component"
    if not m: return ''
    prefix = m[0]
    for item in m:
        for i in range(len(prefix)):
            if prefix[:i+1] <> item[:i+1]:
                prefix = prefix[:i]
                if i == 0: return ''
                break
    return prefix

def islink(path):
    """Test whether a path is a symbolic link"""
    try:
        st = os.lstat(path)
    except (os.error, AttributeError):
        return False
    return stat.S_ISLNK(st.st_mode)

def samefile(path, path2):
    """Test whether two pathnames reference the same actual file"""
    path = _tostr(path, "samefile")
    path2 = _tostr(path2, "samefile")
    return _realpath(path) == _realpath(path2)

def ismount(path):
    """Test whether a path is a mount point.

    XXX This incorrectly always returns false under JDK.

    """
    return 0

def walk(top, func, arg):
    """Walk a directory tree.

    walk(top,func,args) calls func(arg, d, files) for each directory
    "d" in the tree rooted at "top" (including "top" itself).  "files"
    is a list of all the files and subdirs in directory "d".

    """
    try:
        names = os.listdir(top)
    except os.error:
        return
    func(arg, top, names)
    for name in names:
        name = join(top, name)
        if isdir(name) and not islink(name):
            walk(name, func, arg)

def expanduser(path):
    if path[:1] == "~":
        c = path[1:2]
        if not c:
            return gethome()
        if c == os.sep:
            return asPyString(File(gethome(), path[2:]).getPath())
    return path

def getuser():
    return System.getProperty("user.name")

def gethome():
    return System.getProperty("user.home")


# normpath() from Python 1.5.2, with Java appropriate generalizations

# Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A/B.
# It should be understood that this may change the meaning of the path
# if it contains symbolic links!
def normpath(path):
    """Normalize path, eliminating double slashes, etc."""
    sep = os.sep
    if sep == '\\':
        path = path.replace("/", sep)
    curdir = os.curdir
    pardir = os.pardir
    import string
    # Treat initial slashes specially
    slashes = ''
    while path[:1] == sep:
        slashes = slashes + sep
        path = path[1:]
    comps = string.splitfields(path, sep)
    i = 0
    while i < len(comps):
        if comps[i] == curdir:
            del comps[i]
            while i < len(comps) and comps[i] == '':
                del comps[i]
        elif comps[i] == pardir and i > 0 and comps[i-1] not in ('', pardir):
            del comps[i-1:i+1]
            i = i-1
        elif comps[i] == '' and i > 0 and comps[i-1] <> '':
            del comps[i]
        else:
            i = i+1
    # If the path is now empty, substitute '.'
    if not comps and not slashes:
        comps.append(curdir)
    return slashes + string.joinfields(comps, sep)

def abspath(path):
    """Return an absolute path normalized but symbolic links not eliminated"""
    path = _tostr(path, "abspath")
    return _abspath(path)

def _abspath(path):
    # Must use normpath separately because getAbsolutePath doesn't normalize
    # and getCanonicalPath would eliminate symlinks.
    return normpath(asPyString(File(sys.getPath(path)).getAbsolutePath()))

def realpath(path):
    """Return an absolute path normalized and symbolic links eliminated"""
    path = _tostr(path, "realpath")
    return _realpath(path)

def _realpath(path):
    try:
        return asPyString(File(sys.getPath(path)).getCanonicalPath())
    except java.io.IOException:
        return _abspath(path)

def getsize(path):
    path = _tostr(path, "getsize")
    f = File(sys.getPath(path))
    size = f.length()
    # Sadly, if the returned length is zero, we don't really know if the file
    # is zero sized or does not exist.
    if size == 0 and not f.exists():
        raise OSError(0, 'No such file or directory', path)
    return size

def getmtime(path):
    path = _tostr(path, "getmtime")
    f = File(sys.getPath(path))
    if not f.exists():
        raise OSError(0, 'No such file or directory', path)
    return f.lastModified() / 1000.0

def getatime(path):
    # We can't detect access time so we return modification time. This
    # matches the behaviour in os.stat().
    path = _tostr(path, "getatime")
    f = File(sys.getPath(path))
    if not f.exists():
        raise OSError(0, 'No such file or directory', path)
    return f.lastModified() / 1000.0


# expandvars is stolen from CPython-2.1.1's Lib/ntpath.py:

# Expand paths containing shell variable substitutions.
# The following rules apply:
#       - no expansion within single quotes
#       - no escape character, except for '$$' which is translated into '$'
#       - ${varname} is accepted.
#       - varnames can be made out of letters, digits and the character '_'
# XXX With COMMAND.COM you can use any characters in a variable name,
# XXX except '^|<>='.

def expandvars(path):
    """Expand shell variables of form $var and ${var}.

    Unknown variables are left unchanged."""
    if '$' not in path:
        return path
    import string
    varchars = string.letters + string.digits + '_-'
    res = ''
    index = 0
    pathlen = len(path)
    while index < pathlen:
        c = path[index]
        if c == '\'':   # no expansion within single quotes
            path = path[index + 1:]
            pathlen = len(path)
            try:
                index = path.index('\'')
                res = res + '\'' + path[:index + 1]
            except ValueError:
                res = res + path
                index = pathlen - 1
        elif c == '$':  # variable or '$$'
            if path[index + 1:index + 2] == '$':
                res = res + c
                index = index + 1
            elif path[index + 1:index + 2] == '{':
                path = path[index+2:]
                pathlen = len(path)
                try:
                    index = path.index('}')
                    var = path[:index]
                    if os.environ.has_key(var):
                        res = res + os.environ[var]
                except ValueError:
                    res = res + path
                    index = pathlen - 1
            else:
                var = ''
                index = index + 1
                c = path[index:index + 1]
                while c != '' and c in varchars:
                    var = var + c
                    index = index + 1
                    c = path[index:index + 1]
                if os.environ.has_key(var):
                    res = res + os.environ[var]
                if c != '':
                    res = res + c
        else:
            res = res + c
        index = index + 1
    return res
