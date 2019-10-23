"""distutils.dir_util

Utility functions for manipulating directories and directory trees."""

# This module should be kept compatible with Python 2.1.

__revision__ = "$Id: dir_util.py 39416 2005-08-26 15:20:46Z tim_one $"

import os, sys
from types import *
from distutils.errors import DistutilsFileError, DistutilsInternalError
from distutils import log

# cache for by mkpath() -- in addition to cheapening redundant calls,
# eliminates redundant "creating /foo/bar/baz" messages in dry-run mode
_path_created = {}

# I don't use os.makedirs because a) it's new to Python 1.5.2, and
# b) it blows up if the directory already exists (I want to silently
# succeed in that case).
def mkpath (name, mode=0777, verbose=0, dry_run=0):
    """Create a directory and any missing ancestor directories.  If the
       directory already exists (or if 'name' is the empty string, which
       means the current directory, which of course exists), then do
       nothing.  Raise DistutilsFileError if unable to create some
       directory along the way (eg. some sub-path exists, but is a file
       rather than a directory).  If 'verbose' is true, print a one-line
       summary of each mkdir to stdout.  Return the list of directories
       actually created."""

    global _path_created

    # Detect a common bug -- name is None
    if not isinstance(name, StringTypes):
        raise DistutilsInternalError, \
              "mkpath: 'name' must be a string (got %r)" % (name,)

    # XXX what's the better way to handle verbosity? print as we create
    # each directory in the path (the current behaviour), or only announce
    # the creation of the whole path? (quite easy to do the latter since
    # we're not using a recursive algorithm)

    name = os.path.normpath(name)
    created_dirs = []
    if os.path.isdir(name) or name == '':
        return created_dirs
    if _path_created.get(os.path.abspath(name)):
        return created_dirs

    (head, tail) = os.path.split(name)
    tails = [tail]                      # stack of lone dirs to create

    while head and tail and not os.path.isdir(head):
        #print "splitting '%s': " % head,
        (head, tail) = os.path.split(head)
        #print "to ('%s','%s')" % (head, tail)
        tails.insert(0, tail)          # push next higher dir onto stack

    #print "stack of tails:", tails

    # now 'head' contains the deepest directory that already exists
    # (that is, the child of 'head' in 'name' is the highest directory
    # that does *not* exist)
    for d in tails:
        #print "head = %s, d = %s: " % (head, d),
        head = os.path.join(head, d)
        abs_head = os.path.abspath(head)

        if _path_created.get(abs_head):
            continue

        log.info("creating %s", head)

        if not dry_run:
            try:
                os.mkdir(head)
                created_dirs.append(head)
            except OSError, exc:
                raise DistutilsFileError, \
                      "could not create '%s': %s" % (head, exc[-1])

        _path_created[abs_head] = 1
    return created_dirs

# mkpath ()


def create_tree (base_dir, files, mode=0777, verbose=0, dry_run=0):

    """Create all the empty directories under 'base_dir' needed to
       put 'files' there.  'base_dir' is just the a name of a directory
       which doesn't necessarily exist yet; 'files' is a list of filenames
       to be interpreted relative to 'base_dir'.  'base_dir' + the
       directory portion of every file in 'files' will be created if it
       doesn't already exist.  'mode', 'verbose' and 'dry_run' flags are as
       for 'mkpath()'."""

    # First get the list of directories to create
    need_dir = {}
    for file in files:
        need_dir[os.path.join(base_dir, os.path.dirname(file))] = 1
    need_dirs = need_dir.keys()
    need_dirs.sort()

    # Now create them
    for dir in need_dirs:
        mkpath(dir, mode, dry_run=dry_run)

# create_tree ()


def copy_tree (src, dst,
               preserve_mode=1,
               preserve_times=1,
               preserve_symlinks=0,
               update=0,
               verbose=0,
               dry_run=0):

    """Copy an entire directory tree 'src' to a new location 'dst'.  Both
       'src' and 'dst' must be directory names.  If 'src' is not a
       directory, raise DistutilsFileError.  If 'dst' does not exist, it is
       created with 'mkpath()'.  The end result of the copy is that every
       file in 'src' is copied to 'dst', and directories under 'src' are
       recursively copied to 'dst'.  Return the list of files that were
       copied or might have been copied, using their output name.  The
       return value is unaffected by 'update' or 'dry_run': it is simply
       the list of all files under 'src', with the names changed to be
       under 'dst'.

       'preserve_mode' and 'preserve_times' are the same as for
       'copy_file'; note that they only apply to regular files, not to
       directories.  If 'preserve_symlinks' is true, symlinks will be
       copied as symlinks (on platforms that support them!); otherwise
       (the default), the destination of the symlink will be copied.
       'update' and 'verbose' are the same as for 'copy_file'."""

    from distutils.file_util import copy_file

    if not dry_run and not os.path.isdir(src):
        raise DistutilsFileError, \
              "cannot copy tree '%s': not a directory" % src
    try:
        names = os.listdir(src)
    except os.error, (errno, errstr):
        if dry_run:
            names = []
        else:
            raise DistutilsFileError, \
                  "error listing files in '%s': %s" % (src, errstr)

    if not dry_run:
        mkpath(dst)

    outputs = []

    for n in names:
        src_name = os.path.join(src, n)
        dst_name = os.path.join(dst, n)

        if preserve_symlinks and os.path.islink(src_name):
            link_dest = os.readlink(src_name)
            log.info("linking %s -> %s", dst_name, link_dest)
            if not dry_run:
                os.symlink(link_dest, dst_name)
            outputs.append(dst_name)

        elif os.path.isdir(src_name):
            outputs.extend(
                copy_tree(src_name, dst_name, preserve_mode,
                          preserve_times, preserve_symlinks, update,
                          dry_run=dry_run))
        else:
            copy_file(src_name, dst_name, preserve_mode,
                      preserve_times, update, dry_run=dry_run)
            outputs.append(dst_name)

    return outputs

# copy_tree ()

# Helper for remove_tree()
def _build_cmdtuple(path, cmdtuples):
    for f in os.listdir(path):
        real_f = os.path.join(path,f)
        if os.path.isdir(real_f) and not os.path.islink(real_f):
            _build_cmdtuple(real_f, cmdtuples)
        else:
            cmdtuples.append((os.remove, real_f))
    cmdtuples.append((os.rmdir, path))


def remove_tree (directory, verbose=0, dry_run=0):
    """Recursively remove an entire directory tree.  Any errors are ignored
    (apart from being reported to stdout if 'verbose' is true).
    """
    from distutils.util import grok_environment_error
    global _path_created

    log.info("removing '%s' (and everything under it)", directory)
    if dry_run:
        return
    cmdtuples = []
    _build_cmdtuple(directory, cmdtuples)
    for cmd in cmdtuples:
        try:
            apply(cmd[0], (cmd[1],))
            # remove dir from cache if it's already there
            abspath = os.path.abspath(cmd[1])
            if _path_created.has_key(abspath):
                del _path_created[abspath]
        except (IOError, OSError), exc:
            log.warn(grok_environment_error(
                    exc, "error removing %s: " % directory))


def ensure_relative (path):
    """Take the full path 'path', and make it a relative path so
    it can be the second argument to os.path.join().
    """
    drive, path = os.path.splitdrive(path)
    if sys.platform == 'mac':
        return os.sep + path
    else:
        if path[0:1] == os.sep:
            path = drive + path[1:]
        return path
