"""Common operations on Posix pathnames.

Instead of importing this module directly, import os and refer to
this module as os.path.  The "os.path" name is an alias for this
module on Posix systems; on other systems (e.g. Mac, Windows),
os.path provides the same operations in a manner specific to that
platform, and is an alias to another module (e.g. macpath, ntpath).

Some of this can actually be useful on non-Posix systems too, e.g.
for manipulation of the pathname component of URLs.
"""

import java.io.File
import java.io.IOException
import os
import stat
from org.python.core.Py import newString

__all__ = ["normcase","isabs","join","splitdrive","split","splitext",
           "basename","dirname","commonprefix","getsize","getmtime",
           "getatime","getctime","islink","exists","lexists","isdir","isfile",
           "walk","expanduser","expandvars","normpath","abspath",
           "samefile",
           "curdir","pardir","sep","pathsep","defpath","altsep","extsep",
           "devnull","realpath","supports_unicode_filenames"]

# strings representing various path-related bits and pieces
curdir = '.'
pardir = '..'
extsep = '.'
sep = '/'
pathsep = ':'
defpath = ':/bin:/usr/bin'
altsep = None
devnull = '/dev/null'

# Normalize the case of a pathname.  Trivial in Posix, string.lower on Mac.
# On MS-DOS this may also turn slashes into backslashes; however, other
# normalizations (such as optimizing '../' away) are not allowed
# (another function should be defined to do that).

def normcase(s):
    """Normalize case of pathname.  Has no effect under Posix"""
    return s


# Return whether a path is absolute.
# Trivial in Posix, harder on the Mac or MS-DOS.

def isabs(s):
    """Test whether a path is absolute"""
    return s.startswith('/')


# Join pathnames.
# Ignore the previous parts if a part is absolute.
# Insert a '/' unless the first part is empty or already ends in '/'.

def join(a, *p):
    """Join two or more pathname components, inserting '/' as needed"""
    path = a
    for b in p:
        if b.startswith('/'):
            path = b
        elif path == '' or path.endswith('/'):
            path +=  b
        else:
            path += '/' + b
    return path


# Split a path in head (everything up to the last '/') and tail (the
# rest).  If the path ends in '/', tail will be empty.  If there is no
# '/' in the path, head  will be empty.
# Trailing '/'es are stripped from head unless it is the root.

def split(p):
    """Split a pathname.  Returns tuple "(head, tail)" where "tail" is
    everything after the final slash.  Either part may be empty."""
    i = p.rfind('/') + 1
    head, tail = p[:i], p[i:]
    if head and head != '/'*len(head):
        head = head.rstrip('/')
    return head, tail


# Split a path in root and extension.
# The extension is everything starting at the last dot in the last
# pathname component; the root is everything before that.
# It is always true that root + ext == p.

def splitext(p):
    """Split the extension from a pathname.  Extension is everything from the
    last dot to the end.  Returns "(root, ext)", either part may be empty."""
    i = p.rfind('.')
    if i<=p.rfind('/'):
        return p, ''
    else:
        return p[:i], p[i:]


# Split a pathname into a drive specification and the rest of the
# path.  Useful on DOS/Windows/NT; on Unix, the drive is always empty.

def splitdrive(p):
    """Split a pathname into drive and path. On Posix, drive is always
    empty."""
    return '', p


# Return the tail (basename) part of a path.

def basename(p):
    """Returns the final component of a pathname"""
    return split(p)[1]


# Return the head (dirname) part of a path.

def dirname(p):
    """Returns the directory component of a pathname"""
    return split(p)[0]


# Return the longest prefix of all list elements.

def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    n = min(len(s1), len(s2))
    for i in xrange(n):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:n]

# Get size, mtime, atime of files.

def getsize(filename):
    """Return the size of a file, reported by os.stat()."""
    return os.stat(filename).st_size

def getmtime(filename):
    """Return the last modification time of a file, reported by os.stat()."""
    return os.stat(filename).st_mtime

def getatime(filename):
    """Return the last access time of a file, reported by os.stat()."""
    return os.stat(filename).st_atime

def getctime(filename):
    """Return the metadata change time of a file, reported by os.stat()."""
    return os.stat(filename).st_ctime

# Is a path a symbolic link?
# This will always return false on systems where os.lstat doesn't exist.

def islink(path):
    """Test whether a path is a symbolic link"""
    try:
        st = os.lstat(path)
    except (os.error, AttributeError):
        return False
    return stat.S_ISLNK(st.st_mode)


# Does a path exist?
# This is false for dangling symbolic links.

def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True


# Being true for dangling symbolic links is also useful.

def lexists(path):
    """Test whether a path exists.  Returns True for broken symbolic links"""
    try:
        st = os.lstat(path)
    except os.error:
        return False
    return True


# Is a path a directory?
# This follows symbolic links, so both islink() and isdir() can be true
# for the same path.

def isdir(path):
    """Test whether a path is a directory"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return stat.S_ISDIR(st.st_mode)


# Is a path a regular file?
# This follows symbolic links, so both islink() and isfile() can be true
# for the same path.

def isfile(path):
    """Test whether a path is a regular file"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return stat.S_ISREG(st.st_mode)


# Are two filenames really pointing to the same file?

if not os._native_posix:
    def samefile(f1, f2):
        """Test whether two pathnames reference the same actual file"""
        canon1 = newString(java.io.File(_ensure_str(f1)).getCanonicalPath())
        canon2 = newString(java.io.File(_ensure_str(f2)).getCanonicalPath())
        return canon1 == canon2
else:
    def samefile(f1, f2):
        """Test whether two pathnames reference the same actual file"""
        s1 = os.stat(f1)
        s2 = os.stat(f2)
        return samestat(s1, s2)


# XXX: Jython currently lacks fstat
if hasattr(os, 'fstat'):
    # Are two open files really referencing the same file?
    # (Not necessarily the same file descriptor!)

    def sameopenfile(fp1, fp2):
        """Test whether two open file objects reference the same file"""
        s1 = os.fstat(fp1)
        s2 = os.fstat(fp2)
        return samestat(s1, s2)

    __all__.append("sameopenfile")


# XXX: Pure Java stat lacks st_ino/st_dev
if os._native_posix:
    # Are two stat buffers (obtained from stat, fstat or lstat)
    # describing the same file?

    def samestat(s1, s2):
        """Test whether two stat buffers reference the same file"""
        return s1.st_ino == s2.st_ino and \
               s1.st_dev == s2.st_dev


    # Is a path a mount point?
    # (Does this work for all UNIXes?  Is it even guaranteed to work by Posix?)

    def ismount(path):
        """Test whether a path is a mount point"""
        try:
            s1 = os.lstat(path)
            s2 = os.lstat(join(path, '..'))
        except os.error:
            return False # It doesn't exist -- so not a mount point :-)
        dev1 = s1.st_dev
        dev2 = s2.st_dev
        if dev1 != dev2:
            return True     # path/.. on a different device as path
        ino1 = s1.st_ino
        ino2 = s2.st_ino
        if ino1 == ino2:
            return True     # path/.. is the same i-node as path
        return False

    __all__.extend(["samestat", "ismount"])


# Directory tree walk.
# For each directory under top (including top itself, but excluding
# '.' and '..'), func(arg, dirname, filenames) is called, where
# dirname is the name of the directory and filenames is the list
# of files (and subdirectories etc.) in the directory.
# The func may modify the filenames list, to implement a filter,
# or to impose a different order of visiting.

def walk(top, func, arg):
    """Directory tree walk with callback function.

    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), call func(arg, dirname, fnames).
    dirname is the name of the directory, and fnames a list of the names of
    the files and subdirectories in dirname (excluding '.' and '..').  func
    may modify the fnames list in-place (e.g. via del or slice assignment),
    and walk will only recurse into the subdirectories whose names remain in
    fnames; this can be used to implement a filter, or to impose a specific
    order of visiting.  No semantics are defined for, or required of, arg,
    beyond that arg is always passed to func.  It can be used, e.g., to pass
    a filename pattern, or a mutable object designed to accumulate
    statistics.  Passing None for arg is common."""

    try:
        names = os.listdir(top)
    except os.error:
        return
    func(arg, top, names)
    for name in names:
        name = join(top, name)
        try:
            st = os.lstat(name)
        except os.error:
            continue
        if stat.S_ISDIR(st.st_mode):
            walk(name, func, arg)


# Expand paths beginning with '~' or '~user'.
# '~' means $HOME; '~user' means that user's home directory.
# If the path doesn't begin with '~', or if the user or $HOME is unknown,
# the path is returned unchanged (leaving error reporting to whatever
# function is called with the expanded path as argument).
# See also module 'glob' for expansion of *, ? and [...] in pathnames.
# (A function should also be defined to do full *sh-style environment
# variable expansion.)

def expanduser(path):
    """Expand ~ and ~user constructions.  If user or $HOME is unknown,
    do nothing."""
    if not path.startswith('~'):
        return path
    i = path.find('/', 1)
    if i < 0:
        i = len(path)
    if i == 1:
        if 'HOME' not in os.environ:
            return path
        else:
            userhome = os.environ['HOME']
    else:
        # XXX: Jython lacks the pwd module: '~user' isn't supported
        return path
    userhome = userhome.rstrip('/')
    return userhome + path[i:]


# Expand paths containing shell variable substitutions.
# This expands the forms $variable and ${variable} only.
# Non-existent variables are left unchanged.

_varprog = None

def expandvars(path):
    """Expand shell variables of form $var and ${var}.  Unknown variables
    are left unchanged."""
    global _varprog
    if '$' not in path:
        return path
    if not _varprog:
        import re
        _varprog = re.compile(r'\$(\w+|\{[^}]*\})')
    i = 0
    while True:
        m = _varprog.search(path, i)
        if not m:
            break
        i, j = m.span(0)
        name = m.group(1)
        if name.startswith('{') and name.endswith('}'):
            name = name[1:-1]
        if name in os.environ:
            tail = path[j:]
            path = path[:i] + os.environ[name]
            i = len(path)
            path += tail
        else:
            i = j
    return path


# Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A/B.
# It should be understood that this may change the meaning of the path
# if it contains symbolic links!

def normpath(path):
    """Normalize path, eliminating double slashes, etc."""
    if path == '':
        return '.'
    initial_slashes = path.startswith('/')
    # POSIX allows one or two initial slashes, but treats three or more
    # as single slash.
    if (initial_slashes and
        path.startswith('//') and not path.startswith('///')):
        initial_slashes = 2
    comps = path.split('/')
    new_comps = []
    for comp in comps:
        if comp in ('', '.'):
            continue
        if (comp != '..' or (not initial_slashes and not new_comps) or
             (new_comps and new_comps[-1] == '..')):
            new_comps.append(comp)
        elif new_comps:
            new_comps.pop()
    comps = new_comps
    path = '/'.join(comps)
    if initial_slashes:
        path = '/'*initial_slashes + path
    return path or '.'


def abspath(path):
    """Return an absolute path."""
    if not isabs(path):
        path = join(os.getcwd(), path)
    return normpath(path)


# Return a canonical path (i.e. the absolute location of a file on the
# filesystem).

def realpath(filename):
    """Return the canonical path of the specified filename, eliminating any
symbolic links encountered in the path."""
    if isabs(filename):
        bits = ['/'] + filename.split('/')[1:]
    else:
        bits = [''] + filename.split('/')

    for i in range(2, len(bits)+1):
        component = join(*bits[0:i])
        # Resolve symbolic links.
        if islink(component):
            resolved = _resolve_link(component)
            if resolved is None:
                # Infinite loop -- return original component + rest of the path
                return abspath(join(*([component] + bits[i:])))
            else:
                newpath = join(*([resolved] + bits[i:]))
                return realpath(newpath)

    return abspath(filename)


if not os._native_posix:
    def _resolve_link(path):
        """Internal helper function.  Takes a path and follows symlinks
        until we either arrive at something that isn't a symlink, or
        encounter a path we've seen before (meaning that there's a loop).
        """
        try:
            return newString(java.io.File(abspath(path)).getCanonicalPath())
        except java.io.IOException:
            return None
else:
    def _resolve_link(path):
        """Internal helper function.  Takes a path and follows symlinks
        until we either arrive at something that isn't a symlink, or
        encounter a path we've seen before (meaning that there's a loop).
        """
        paths_seen = []
        while islink(path):
            if path in paths_seen:
                # Already seen this path, so we must have a symlink loop
                return None
            paths_seen.append(path)
            # Resolve where the link points to
            resolved = os.readlink(path)
            if not isabs(resolved):
                dir = dirname(path)
                path = normpath(join(dir, resolved))
            else:
                path = normpath(resolved)
        return path


def _ensure_str(obj):
    """Ensure obj is a string, otherwise raise a TypeError"""
    if isinstance(obj, basestring):
        return obj
    raise TypeError('coercing to Unicode: need string or buffer, %s found' % \
                        _type_name(obj))


def _type_name(obj):
    """Determine the appropriate type name of obj for display"""
    TPFLAGS_HEAPTYPE = 1 << 9
    type_name = ''
    obj_type = type(obj)
    is_heap = obj_type.__flags__ & TPFLAGS_HEAPTYPE == TPFLAGS_HEAPTYPE
    if not is_heap and obj_type.__module__ != '__builtin__':
        type_name = '%s.' % obj_type.__module__
    type_name += obj_type.__name__
    return type_name

supports_unicode_filenames = False
