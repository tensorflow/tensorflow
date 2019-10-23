"""Helper class to quickly write a loop over all standard input files.

Typical use is:

    import fileinput
    for line in fileinput.input():
        process(line)

This iterates over the lines of all files listed in sys.argv[1:],
defaulting to sys.stdin if the list is empty.  If a filename is '-' it
is also replaced by sys.stdin.  To specify an alternative list of
filenames, pass it as the argument to input().  A single file name is
also allowed.

Functions filename(), lineno() return the filename and cumulative line
number of the line that has just been read; filelineno() returns its
line number in the current file; isfirstline() returns true iff the
line just read is the first line of its file; isstdin() returns true
iff the line was read from sys.stdin.  Function nextfile() closes the
current file so that the next iteration will read the first line from
the next file (if any); lines not read from the file will not count
towards the cumulative line count; the filename is not changed until
after the first line of the next file has been read.  Function close()
closes the sequence.

Before any lines have been read, filename() returns None and both line
numbers are zero; nextfile() has no effect.  After all lines have been
read, filename() and the line number functions return the values
pertaining to the last line read; nextfile() has no effect.

All files are opened in text mode by default, you can override this by
setting the mode parameter to input() or FileInput.__init__().
If an I/O error occurs during opening or reading a file, the IOError
exception is raised.

If sys.stdin is used more than once, the second and further use will
return no lines, except perhaps for interactive use, or if it has been
explicitly reset (e.g. using sys.stdin.seek(0)).

Empty files are opened and immediately closed; the only time their
presence in the list of filenames is noticeable at all is when the
last file opened is empty.

It is possible that the last line of a file doesn't end in a newline
character; otherwise lines are returned including the trailing
newline.

Class FileInput is the implementation; its methods filename(),
lineno(), fileline(), isfirstline(), isstdin(), nextfile() and close()
correspond to the functions in the module.  In addition it has a
readline() method which returns the next input line, and a
__getitem__() method which implements the sequence behavior.  The
sequence must be accessed in strictly sequential order; sequence
access and readline() cannot be mixed.

Optional in-place filtering: if the keyword argument inplace=1 is
passed to input() or to the FileInput constructor, the file is moved
to a backup file and standard output is directed to the input file.
This makes it possible to write a filter that rewrites its input file
in place.  If the keyword argument backup=".<some extension>" is also
given, it specifies the extension for the backup file, and the backup
file remains around; by default, the extension is ".bak" and it is
deleted when the output file is closed.  In-place filtering is
disabled when standard input is read.  XXX The current implementation
does not work for MS-DOS 8+3 filesystems.

Performance: this module is unfortunately one of the slower ways of
processing large numbers of input lines.  Nevertheless, a significant
speed-up has been obtained by using readlines(bufsize) instead of
readline().  A new keyword argument, bufsize=N, is present on the
input() function and the FileInput() class to override the default
buffer size.

XXX Possible additions:

- optional getopt argument processing
- isatty()
- read(), read(size), even readlines()

"""

import sys, os

__all__ = ["input","close","nextfile","filename","lineno","filelineno",
           "isfirstline","isstdin","FileInput"]

_state = None

DEFAULT_BUFSIZE = 8*1024

def input(files=None, inplace=0, backup="", bufsize=0,
          mode="r", openhook=None):
    """input([files[, inplace[, backup[, mode[, openhook]]]]])

    Create an instance of the FileInput class. The instance will be used
    as global state for the functions of this module, and is also returned
    to use during iteration. The parameters to this function will be passed
    along to the constructor of the FileInput class.
    """
    global _state
    if _state and _state._file:
        raise RuntimeError, "input() already active"
    _state = FileInput(files, inplace, backup, bufsize, mode, openhook)
    return _state

def close():
    """Close the sequence."""
    global _state
    state = _state
    _state = None
    if state:
        state.close()

def nextfile():
    """
    Close the current file so that the next iteration will read the first
    line from the next file (if any); lines not read from the file will
    not count towards the cumulative line count. The filename is not
    changed until after the first line of the next file has been read.
    Before the first line has been read, this function has no effect;
    it cannot be used to skip the first file. After the last line of the
    last file has been read, this function has no effect.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.nextfile()

def filename():
    """
    Return the name of the file currently being read.
    Before the first line has been read, returns None.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.filename()

def lineno():
    """
    Return the cumulative line number of the line that has just been read.
    Before the first line has been read, returns 0. After the last line
    of the last file has been read, returns the line number of that line.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.lineno()

def filelineno():
    """
    Return the line number in the current file. Before the first line
    has been read, returns 0. After the last line of the last file has
    been read, returns the line number of that line within the file.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.filelineno()

def fileno():
    """
    Return the file number of the current file. When no file is currently
    opened, returns -1.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.fileno()

def isfirstline():
    """
    Returns true the line just read is the first line of its file,
    otherwise returns false.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.isfirstline()

def isstdin():
    """
    Returns true if the last line was read from sys.stdin,
    otherwise returns false.
    """
    if not _state:
        raise RuntimeError, "no active input()"
    return _state.isstdin()

class FileInput:
    """class FileInput([files[, inplace[, backup[, mode[, openhook]]]]])

    Class FileInput is the implementation of the module; its methods
    filename(), lineno(), fileline(), isfirstline(), isstdin(), fileno(),
    nextfile() and close() correspond to the functions of the same name
    in the module.
    In addition it has a readline() method which returns the next
    input line, and a __getitem__() method which implements the
    sequence behavior. The sequence must be accessed in strictly
    sequential order; random access and readline() cannot be mixed.
    """

    def __init__(self, files=None, inplace=0, backup="", bufsize=0,
                 mode="r", openhook=None):
        if isinstance(files, basestring):
            files = (files,)
        else:
            if files is None:
                files = sys.argv[1:]
            if not files:
                files = ('-',)
            else:
                files = tuple(files)
        self._files = files
        self._inplace = inplace
        self._backup = backup
        self._bufsize = bufsize or DEFAULT_BUFSIZE
        self._savestdout = None
        self._output = None
        self._filename = None
        self._lineno = 0
        self._filelineno = 0
        self._file = None
        self._isstdin = False
        self._backupfilename = None
        self._buffer = []
        self._bufindex = 0
        # restrict mode argument to reading modes
        if mode not in ('r', 'rU', 'U', 'rb'):
            raise ValueError("FileInput opening mode must be one of "
                             "'r', 'rU', 'U' and 'rb'")
        self._mode = mode
        if inplace and openhook:
            raise ValueError("FileInput cannot use an opening hook in inplace mode")
        elif openhook and not callable(openhook):
            raise ValueError("FileInput openhook must be callable")
        self._openhook = openhook

    def __del__(self):
        self.close()

    def close(self):
        self.nextfile()
        self._files = ()

    def __iter__(self):
        return self

    def next(self):
        try:
            line = self._buffer[self._bufindex]
        except IndexError:
            pass
        else:
            self._bufindex += 1
            self._lineno += 1
            self._filelineno += 1
            return line
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __getitem__(self, i):
        if i != self._lineno:
            raise RuntimeError, "accessing lines out of order"
        try:
            return self.next()
        except StopIteration:
            raise IndexError, "end of input reached"

    def nextfile(self):
        savestdout = self._savestdout
        self._savestdout = 0
        if savestdout:
            sys.stdout = savestdout

        output = self._output
        self._output = 0
        if output:
            output.close()

        file = self._file
        self._file = 0
        if file and not self._isstdin:
            file.close()

        backupfilename = self._backupfilename
        self._backupfilename = 0
        if backupfilename and not self._backup:
            try: os.unlink(backupfilename)
            except OSError: pass

        self._isstdin = False
        self._buffer = []
        self._bufindex = 0

    def readline(self):
        try:
            line = self._buffer[self._bufindex]
        except IndexError:
            pass
        else:
            self._bufindex += 1
            self._lineno += 1
            self._filelineno += 1
            return line
        if not self._file:
            if not self._files:
                return ""
            self._filename = self._files[0]
            self._files = self._files[1:]
            self._filelineno = 0
            self._file = None
            self._isstdin = False
            self._backupfilename = 0
            if self._filename == '-':
                self._filename = '<stdin>'
                self._file = sys.stdin
                self._isstdin = True
            else:
                if self._inplace:
                    self._backupfilename = (
                        self._filename + (self._backup or os.extsep+"bak"))
                    try: os.unlink(self._backupfilename)
                    except os.error: pass
                    # The next few lines may raise IOError
                    os.rename(self._filename, self._backupfilename)
                    self._file = open(self._backupfilename, self._mode)
                    try:
                        perm = os.fstat(self._file.fileno()).st_mode
                    except (AttributeError, OSError):
                        # AttributeError occurs in Jython, where there's no
                        # os.fstat.
                        self._output = open(self._filename, "w")
                    else:
                        fd = os.open(self._filename,
                                     os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                                     perm)
                        self._output = os.fdopen(fd, "w")
                        try:
                            if hasattr(os, 'chmod'):
                                os.chmod(self._filename, perm)
                        except OSError:
                            pass
                    self._savestdout = sys.stdout
                    sys.stdout = self._output
                else:
                    # This may raise IOError
                    if self._openhook:
                        self._file = self._openhook(self._filename, self._mode)
                    else:
                        self._file = open(self._filename, self._mode)
        self._buffer = self._file.readlines(self._bufsize)
        self._bufindex = 0
        if not self._buffer:
            self.nextfile()
        # Recursive call
        return self.readline()

    def filename(self):
        return self._filename

    def lineno(self):
        return self._lineno

    def filelineno(self):
        return self._filelineno

    def fileno(self):
        if self._file:
            try:
                return self._file.fileno()
            except ValueError:
                return -1
        else:
            return -1

    def isfirstline(self):
        return self._filelineno == 1

    def isstdin(self):
        return self._isstdin


def hook_compressed(filename, mode):
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        return gzip.open(filename, mode)
    elif ext == '.bz2':
        import bz2
        return bz2.BZ2File(filename, mode)
    else:
        return open(filename, mode)


def hook_encoded(encoding):
    import codecs
    def openhook(filename, mode):
        return codecs.open(filename, mode, encoding)
    return openhook


def _test():
    import getopt
    inplace = 0
    backup = 0
    opts, args = getopt.getopt(sys.argv[1:], "ib:")
    for o, a in opts:
        if o == '-i': inplace = 1
        if o == '-b': backup = a
    for line in input(args, inplace=inplace, backup=backup):
        if line[-1:] == '\n': line = line[:-1]
        if line[-1:] == '\r': line = line[:-1]
        print "%d: %s[%d]%s %s" % (lineno(), filename(), filelineno(),
                                   isfirstline() and "*" or "", line)
    print "%d: %s[%d]" % (lineno(), filename(), filelineno())

if __name__ == '__main__':
    _test()
