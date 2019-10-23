#! /usr/local/bin/python

# NOTE: the above "/usr/local/bin/python" is NOT a mistake.  It is
# intentionally NOT "/usr/bin/env python".  On many systems
# (e.g. Solaris), /usr/local/bin is not in $PATH as passed to CGI
# scripts, and /usr/local/bin is the default directory where Python is
# installed, so /usr/bin/env would be unable to find python.  Granted,
# binary installations by Linux vendors often install Python in
# /usr/bin.  So let those vendors patch cgi.py to match their choice
# of installation.

"""Support module for CGI (Common Gateway Interface) scripts.

This module defines a number of utilities for use by CGI scripts
written in Python.
"""

# XXX Perhaps there should be a slimmed version that doesn't contain
# all those backwards compatible and debugging classes and functions?

# History
# -------
#
# Michael McLay started this module.  Steve Majewski changed the
# interface to SvFormContentDict and FormContentDict.  The multipart
# parsing was inspired by code submitted by Andreas Paepcke.  Guido van
# Rossum rewrote, reformatted and documented the module and is currently
# responsible for its maintenance.
#

__version__ = "2.6"


# Imports
# =======

from operator import attrgetter
import sys
import os
import urllib
import mimetools
import rfc822
import UserDict
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

__all__ = ["MiniFieldStorage", "FieldStorage", "FormContentDict",
           "SvFormContentDict", "InterpFormContentDict", "FormContent",
           "parse", "parse_qs", "parse_qsl", "parse_multipart",
           "parse_header", "print_exception", "print_environ",
           "print_form", "print_directory", "print_arguments",
           "print_environ_usage", "escape"]

# Logging support
# ===============

logfile = ""            # Filename to log to, if not empty
logfp = None            # File object to log to, if not None

def initlog(*allargs):
    """Write a log message, if there is a log file.

    Even though this function is called initlog(), you should always
    use log(); log is a variable that is set either to initlog
    (initially), to dolog (once the log file has been opened), or to
    nolog (when logging is disabled).

    The first argument is a format string; the remaining arguments (if
    any) are arguments to the % operator, so e.g.
        log("%s: %s", "a", "b")
    will write "a: b" to the log file, followed by a newline.

    If the global logfp is not None, it should be a file object to
    which log data is written.

    If the global logfp is None, the global logfile may be a string
    giving a filename to open, in append mode.  This file should be
    world writable!!!  If the file can't be opened, logging is
    silently disabled (since there is no safe place where we could
    send an error message).

    """
    global logfp, log
    if logfile and not logfp:
        try:
            logfp = open(logfile, "a")
        except IOError:
            pass
    if not logfp:
        log = nolog
    else:
        log = dolog
    log(*allargs)

def dolog(fmt, *args):
    """Write a log message to the log file.  See initlog() for docs."""
    logfp.write(fmt%args + "\n")

def nolog(*allargs):
    """Dummy function, assigned to log when logging is disabled."""
    pass

log = initlog           # The current logging function


# Parsing functions
# =================

# Maximum input we will accept when REQUEST_METHOD is POST
# 0 ==> unlimited input
maxlen = 0

def parse(fp=None, environ=os.environ, keep_blank_values=0, strict_parsing=0):
    """Parse a query in the environment or from a file (default stdin)

        Arguments, all optional:

        fp              : file pointer; default: sys.stdin

        environ         : environment dictionary; default: os.environ

        keep_blank_values: flag indicating whether blank values in
            URL encoded forms should be treated as blank strings.
            A true value indicates that blanks should be retained as
            blank strings.  The default false value indicates that
            blank values are to be ignored and treated as if they were
            not included.

        strict_parsing: flag indicating what to do with parsing errors.
            If false (the default), errors are silently ignored.
            If true, errors raise a ValueError exception.
    """
    if fp is None:
        fp = sys.stdin
    if not 'REQUEST_METHOD' in environ:
        environ['REQUEST_METHOD'] = 'GET'       # For testing stand-alone
    if environ['REQUEST_METHOD'] == 'POST':
        ctype, pdict = parse_header(environ['CONTENT_TYPE'])
        if ctype == 'multipart/form-data':
            return parse_multipart(fp, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            clength = int(environ['CONTENT_LENGTH'])
            if maxlen and clength > maxlen:
                raise ValueError, 'Maximum content length exceeded'
            qs = fp.read(clength)
        else:
            qs = ''                     # Unknown content-type
        if 'QUERY_STRING' in environ:
            if qs: qs = qs + '&'
            qs = qs + environ['QUERY_STRING']
        elif sys.argv[1:]:
            if qs: qs = qs + '&'
            qs = qs + sys.argv[1]
        environ['QUERY_STRING'] = qs    # XXX Shouldn't, really
    elif 'QUERY_STRING' in environ:
        qs = environ['QUERY_STRING']
    else:
        if sys.argv[1:]:
            qs = sys.argv[1]
        else:
            qs = ""
        environ['QUERY_STRING'] = qs    # XXX Shouldn't, really
    return parse_qs(qs, keep_blank_values, strict_parsing)


def parse_qs(qs, keep_blank_values=0, strict_parsing=0):
    """Parse a query given as a string argument.

        Arguments:

        qs: URL-encoded query string to be parsed

        keep_blank_values: flag indicating whether blank values in
            URL encoded queries should be treated as blank strings.
            A true value indicates that blanks should be retained as
            blank strings.  The default false value indicates that
            blank values are to be ignored and treated as if they were
            not included.

        strict_parsing: flag indicating what to do with parsing errors.
            If false (the default), errors are silently ignored.
            If true, errors raise a ValueError exception.
    """
    dict = {}
    for name, value in parse_qsl(qs, keep_blank_values, strict_parsing):
        if name in dict:
            dict[name].append(value)
        else:
            dict[name] = [value]
    return dict

def parse_qsl(qs, keep_blank_values=0, strict_parsing=0):
    """Parse a query given as a string argument.

    Arguments:

    qs: URL-encoded query string to be parsed

    keep_blank_values: flag indicating whether blank values in
        URL encoded queries should be treated as blank strings.  A
        true value indicates that blanks should be retained as blank
        strings.  The default false value indicates that blank values
        are to be ignored and treated as if they were  not included.

    strict_parsing: flag indicating what to do with parsing errors. If
        false (the default), errors are silently ignored. If true,
        errors raise a ValueError exception.

    Returns a list, as G-d intended.
    """
    pairs = [s2 for s1 in qs.split('&') for s2 in s1.split(';')]
    r = []
    for name_value in pairs:
        if not name_value and not strict_parsing:
            continue
        nv = name_value.split('=', 1)
        if len(nv) != 2:
            if strict_parsing:
                raise ValueError, "bad query field: %r" % (name_value,)
            # Handle case of a control-name with no equal sign
            if keep_blank_values:
                nv.append('')
            else:
                continue
        if len(nv[1]) or keep_blank_values:
            name = urllib.unquote(nv[0].replace('+', ' '))
            value = urllib.unquote(nv[1].replace('+', ' '))
            r.append((name, value))

    return r


def parse_multipart(fp, pdict):
    """Parse multipart input.

    Arguments:
    fp   : input file
    pdict: dictionary containing other parameters of content-type header

    Returns a dictionary just like parse_qs(): keys are the field names, each
    value is a list of values for that field.  This is easy to use but not
    much good if you are expecting megabytes to be uploaded -- in that case,
    use the FieldStorage class instead which is much more flexible.  Note
    that content-type is the raw, unparsed contents of the content-type
    header.

    XXX This does not parse nested multipart parts -- use FieldStorage for
    that.

    XXX This should really be subsumed by FieldStorage altogether -- no
    point in having two implementations of the same parsing algorithm.
    Also, FieldStorage protects itself better against certain DoS attacks
    by limiting the size of the data read in one chunk.  The API here
    does not support that kind of protection.  This also affects parse()
    since it can call parse_multipart().

    """
    boundary = ""
    if 'boundary' in pdict:
        boundary = pdict['boundary']
    if not valid_boundary(boundary):
        raise ValueError,  ('Invalid boundary in multipart form: %r'
                            % (boundary,))

    nextpart = "--" + boundary
    lastpart = "--" + boundary + "--"
    partdict = {}
    terminator = ""

    while terminator != lastpart:
        bytes = -1
        data = None
        if terminator:
            # At start of next part.  Read headers first.
            headers = mimetools.Message(fp)
            clength = headers.getheader('content-length')
            if clength:
                try:
                    bytes = int(clength)
                except ValueError:
                    pass
            if bytes > 0:
                if maxlen and bytes > maxlen:
                    raise ValueError, 'Maximum content length exceeded'
                data = fp.read(bytes)
            else:
                data = ""
        # Read lines until end of part.
        lines = []
        while 1:
            line = fp.readline()
            if not line:
                terminator = lastpart # End outer loop
                break
            if line[:2] == "--":
                terminator = line.strip()
                if terminator in (nextpart, lastpart):
                    break
            lines.append(line)
        # Done with part.
        if data is None:
            continue
        if bytes < 0:
            if lines:
                # Strip final line terminator
                line = lines[-1]
                if line[-2:] == "\r\n":
                    line = line[:-2]
                elif line[-1:] == "\n":
                    line = line[:-1]
                lines[-1] = line
                data = "".join(lines)
        line = headers['content-disposition']
        if not line:
            continue
        key, params = parse_header(line)
        if key != 'form-data':
            continue
        if 'name' in params:
            name = params['name']
        else:
            continue
        if name in partdict:
            partdict[name].append(data)
        else:
            partdict[name] = [data]

    return partdict


def parse_header(line):
    """Parse a Content-type like header.

    Return the main content-type and a dictionary of options.

    """
    plist = [x.strip() for x in line.split(';')]
    key = plist.pop(0).lower()
    pdict = {}
    for p in plist:
        i = p.find('=')
        if i >= 0:
            name = p[:i].strip().lower()
            value = p[i+1:].strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
                value = value.replace('\\\\', '\\').replace('\\"', '"')
            pdict[name] = value
    return key, pdict


# Classes for field storage
# =========================

class MiniFieldStorage:

    """Like FieldStorage, for use when no file uploads are possible."""

    # Dummy attributes
    filename = None
    list = None
    type = None
    file = None
    type_options = {}
    disposition = None
    disposition_options = {}
    headers = {}

    def __init__(self, name, value):
        """Constructor from field name and value."""
        self.name = name
        self.value = value
        # self.file = StringIO(value)

    def __repr__(self):
        """Return printable representation."""
        return "MiniFieldStorage(%r, %r)" % (self.name, self.value)


class FieldStorage:

    """Store a sequence of fields, reading multipart/form-data.

    This class provides naming, typing, files stored on disk, and
    more.  At the top level, it is accessible like a dictionary, whose
    keys are the field names.  (Note: None can occur as a field name.)
    The items are either a Python list (if there's multiple values) or
    another FieldStorage or MiniFieldStorage object.  If it's a single
    object, it has the following attributes:

    name: the field name, if specified; otherwise None

    filename: the filename, if specified; otherwise None; this is the
        client side filename, *not* the file name on which it is
        stored (that's a temporary file you don't deal with)

    value: the value as a *string*; for file uploads, this
        transparently reads the file every time you request the value

    file: the file(-like) object from which you can read the data;
        None if the data is stored a simple string

    type: the content-type, or None if not specified

    type_options: dictionary of options specified on the content-type
        line

    disposition: content-disposition, or None if not specified

    disposition_options: dictionary of corresponding options

    headers: a dictionary(-like) object (sometimes rfc822.Message or a
        subclass thereof) containing *all* headers

    The class is subclassable, mostly for the purpose of overriding
    the make_file() method, which is called internally to come up with
    a file open for reading and writing.  This makes it possible to
    override the default choice of storing all files in a temporary
    directory and unlinking them as soon as they have been opened.

    """

    def __init__(self, fp=None, headers=None, outerboundary="",
                 environ=os.environ, keep_blank_values=0, strict_parsing=0):
        """Constructor.  Read multipart/* until last part.

        Arguments, all optional:

        fp              : file pointer; default: sys.stdin
            (not used when the request method is GET)

        headers         : header dictionary-like object; default:
            taken from environ as per CGI spec

        outerboundary   : terminating multipart boundary
            (for internal use only)

        environ         : environment dictionary; default: os.environ

        keep_blank_values: flag indicating whether blank values in
            URL encoded forms should be treated as blank strings.
            A true value indicates that blanks should be retained as
            blank strings.  The default false value indicates that
            blank values are to be ignored and treated as if they were
            not included.

        strict_parsing: flag indicating what to do with parsing errors.
            If false (the default), errors are silently ignored.
            If true, errors raise a ValueError exception.

        """
        method = 'GET'
        self.keep_blank_values = keep_blank_values
        self.strict_parsing = strict_parsing
        if 'REQUEST_METHOD' in environ:
            method = environ['REQUEST_METHOD'].upper()
        if method == 'GET' or method == 'HEAD':
            if 'QUERY_STRING' in environ:
                qs = environ['QUERY_STRING']
            elif sys.argv[1:]:
                qs = sys.argv[1]
            else:
                qs = ""
            fp = StringIO(qs)
            if headers is None:
                headers = {'content-type':
                           "application/x-www-form-urlencoded"}
        if headers is None:
            headers = {}
            if method == 'POST':
                # Set default content-type for POST to what's traditional
                headers['content-type'] = "application/x-www-form-urlencoded"
            if 'CONTENT_TYPE' in environ:
                headers['content-type'] = environ['CONTENT_TYPE']
            if 'CONTENT_LENGTH' in environ:
                headers['content-length'] = environ['CONTENT_LENGTH']
        self.fp = fp or sys.stdin
        self.headers = headers
        self.outerboundary = outerboundary

        # Process content-disposition header
        cdisp, pdict = "", {}
        if 'content-disposition' in self.headers:
            cdisp, pdict = parse_header(self.headers['content-disposition'])
        self.disposition = cdisp
        self.disposition_options = pdict
        self.name = None
        if 'name' in pdict:
            self.name = pdict['name']
        self.filename = None
        if 'filename' in pdict:
            self.filename = pdict['filename']

        # Process content-type header
        #
        # Honor any existing content-type header.  But if there is no
        # content-type header, use some sensible defaults.  Assume
        # outerboundary is "" at the outer level, but something non-false
        # inside a multi-part.  The default for an inner part is text/plain,
        # but for an outer part it should be urlencoded.  This should catch
        # bogus clients which erroneously forget to include a content-type
        # header.
        #
        # See below for what we do if there does exist a content-type header,
        # but it happens to be something we don't understand.
        if 'content-type' in self.headers:
            ctype, pdict = parse_header(self.headers['content-type'])
        elif self.outerboundary or method != 'POST':
            ctype, pdict = "text/plain", {}
        else:
            ctype, pdict = 'application/x-www-form-urlencoded', {}
        self.type = ctype
        self.type_options = pdict
        self.innerboundary = ""
        if 'boundary' in pdict:
            self.innerboundary = pdict['boundary']
        clen = -1
        if 'content-length' in self.headers:
            try:
                clen = int(self.headers['content-length'])
            except ValueError:
                pass
            if maxlen and clen > maxlen:
                raise ValueError, 'Maximum content length exceeded'
        self.length = clen

        self.list = self.file = None
        self.done = 0
        if ctype == 'application/x-www-form-urlencoded':
            self.read_urlencoded()
        elif ctype[:10] == 'multipart/':
            self.read_multi(environ, keep_blank_values, strict_parsing)
        else:
            self.read_single()

    def __repr__(self):
        """Return a printable representation."""
        return "FieldStorage(%r, %r, %r)" % (
                self.name, self.filename, self.value)

    def __iter__(self):
        return iter(self.keys())

    def __getattr__(self, name):
        if name != 'value':
            raise AttributeError, name
        if self.file:
            self.file.seek(0)
            value = self.file.read()
            self.file.seek(0)
        elif self.list is not None:
            value = self.list
        else:
            value = None
        return value

    def __getitem__(self, key):
        """Dictionary style indexing."""
        if self.list is None:
            raise TypeError, "not indexable"
        found = []
        for item in self.list:
            if item.name == key: found.append(item)
        if not found:
            raise KeyError, key
        if len(found) == 1:
            return found[0]
        else:
            return found

    def getvalue(self, key, default=None):
        """Dictionary style get() method, including 'value' lookup."""
        if key in self:
            value = self[key]
            if type(value) is type([]):
                return map(attrgetter('value'), value)
            else:
                return value.value
        else:
            return default

    def getfirst(self, key, default=None):
        """ Return the first value received."""
        if key in self:
            value = self[key]
            if type(value) is type([]):
                return value[0].value
            else:
                return value.value
        else:
            return default

    def getlist(self, key):
        """ Return list of received values."""
        if key in self:
            value = self[key]
            if type(value) is type([]):
                return map(attrgetter('value'), value)
            else:
                return [value.value]
        else:
            return []

    def keys(self):
        """Dictionary style keys() method."""
        if self.list is None:
            raise TypeError, "not indexable"
        keys = []
        for item in self.list:
            if item.name not in keys: keys.append(item.name)
        return keys

    def has_key(self, key):
        """Dictionary style has_key() method."""
        if self.list is None:
            raise TypeError, "not indexable"
        for item in self.list:
            if item.name == key: return True
        return False

    def __contains__(self, key):
        """Dictionary style __contains__ method."""
        if self.list is None:
            raise TypeError, "not indexable"
        for item in self.list:
            if item.name == key: return True
        return False

    def __len__(self):
        """Dictionary style len(x) support."""
        return len(self.keys())

    def read_urlencoded(self):
        """Internal: read data in query string format."""
        qs = self.fp.read(self.length)
        self.list = list = []
        for key, value in parse_qsl(qs, self.keep_blank_values,
                                    self.strict_parsing):
            list.append(MiniFieldStorage(key, value))
        self.skip_lines()

    FieldStorageClass = None

    def read_multi(self, environ, keep_blank_values, strict_parsing):
        """Internal: read a part that is itself multipart."""
        ib = self.innerboundary
        if not valid_boundary(ib):
            raise ValueError, 'Invalid boundary in multipart form: %r' % (ib,)
        self.list = []
        klass = self.FieldStorageClass or self.__class__
        part = klass(self.fp, {}, ib,
                     environ, keep_blank_values, strict_parsing)
        # Throw first part away
        while not part.done:
            headers = rfc822.Message(self.fp)
            part = klass(self.fp, headers, ib,
                         environ, keep_blank_values, strict_parsing)
            self.list.append(part)
        self.skip_lines()

    def read_single(self):
        """Internal: read an atomic part."""
        if self.length >= 0:
            self.read_binary()
            self.skip_lines()
        else:
            self.read_lines()
        self.file.seek(0)

    bufsize = 8*1024            # I/O buffering size for copy to file

    def read_binary(self):
        """Internal: read binary data."""
        self.file = self.make_file('b')
        todo = self.length
        if todo >= 0:
            while todo > 0:
                data = self.fp.read(min(todo, self.bufsize))
                if not data:
                    self.done = -1
                    break
                self.file.write(data)
                todo = todo - len(data)

    def read_lines(self):
        """Internal: read lines until EOF or outerboundary."""
        self.file = self.__file = StringIO()
        if self.outerboundary:
            self.read_lines_to_outerboundary()
        else:
            self.read_lines_to_eof()

    def __write(self, line):
        if self.__file is not None:
            if self.__file.tell() + len(line) > 1000:
                self.file = self.make_file('')
                self.file.write(self.__file.getvalue())
                self.__file = None
        self.file.write(line)

    def read_lines_to_eof(self):
        """Internal: read lines until EOF."""
        while 1:
            line = self.fp.readline(1<<16)
            if not line:
                self.done = -1
                break
            self.__write(line)

    def read_lines_to_outerboundary(self):
        """Internal: read lines until outerboundary."""
        next = "--" + self.outerboundary
        last = next + "--"
        delim = ""
        last_line_lfend = True
        while 1:
            line = self.fp.readline(1<<16)
            if not line:
                self.done = -1
                break
            if line[:2] == "--" and last_line_lfend:
                strippedline = line.strip()
                if strippedline == next:
                    break
                if strippedline == last:
                    self.done = 1
                    break
            odelim = delim
            if line[-2:] == "\r\n":
                delim = "\r\n"
                line = line[:-2]
                last_line_lfend = True
            elif line[-1] == "\n":
                delim = "\n"
                line = line[:-1]
                last_line_lfend = True
            else:
                delim = ""
                last_line_lfend = False
            self.__write(odelim + line)

    def skip_lines(self):
        """Internal: skip lines until outer boundary if defined."""
        if not self.outerboundary or self.done:
            return
        next = "--" + self.outerboundary
        last = next + "--"
        last_line_lfend = True
        while 1:
            line = self.fp.readline(1<<16)
            if not line:
                self.done = -1
                break
            if line[:2] == "--" and last_line_lfend:
                strippedline = line.strip()
                if strippedline == next:
                    break
                if strippedline == last:
                    self.done = 1
                    break
            last_line_lfend = line.endswith('\n')

    def make_file(self, binary=None):
        """Overridable: return a readable & writable file.

        The file will be used as follows:
        - data is written to it
        - seek(0)
        - data is read from it

        The 'binary' argument is unused -- the file is always opened
        in binary mode.

        This version opens a temporary file for reading and writing,
        and immediately deletes (unlinks) it.  The trick (on Unix!) is
        that the file can still be used, but it can't be opened by
        another process, and it will automatically be deleted when it
        is closed or when the current process terminates.

        If you want a more permanent file, you derive a class which
        overrides this method.  If you want a visible temporary file
        that is nevertheless automatically deleted when the script
        terminates, try defining a __del__ method in a derived class
        which unlinks the temporary files you have created.

        """
        import tempfile
        return tempfile.TemporaryFile("w+b")



# Backwards Compatibility Classes
# ===============================

class FormContentDict(UserDict.UserDict):
    """Form content as dictionary with a list of values per field.

    form = FormContentDict()

    form[key] -> [value, value, ...]
    key in form -> Boolean
    form.keys() -> [key, key, ...]
    form.values() -> [[val, val, ...], [val, val, ...], ...]
    form.items() ->  [(key, [val, val, ...]), (key, [val, val, ...]), ...]
    form.dict == {key: [val, val, ...], ...}

    """
    def __init__(self, environ=os.environ):
        self.dict = self.data = parse(environ=environ)
        self.query_string = environ['QUERY_STRING']


class SvFormContentDict(FormContentDict):
    """Form content as dictionary expecting a single value per field.

    If you only expect a single value for each field, then form[key]
    will return that single value.  It will raise an IndexError if
    that expectation is not true.  If you expect a field to have
    possible multiple values, than you can use form.getlist(key) to
    get all of the values.  values() and items() are a compromise:
    they return single strings where there is a single value, and
    lists of strings otherwise.

    """
    def __getitem__(self, key):
        if len(self.dict[key]) > 1:
            raise IndexError, 'expecting a single value'
        return self.dict[key][0]
    def getlist(self, key):
        return self.dict[key]
    def values(self):
        result = []
        for value in self.dict.values():
            if len(value) == 1:
                result.append(value[0])
            else: result.append(value)
        return result
    def items(self):
        result = []
        for key, value in self.dict.items():
            if len(value) == 1:
                result.append((key, value[0]))
            else: result.append((key, value))
        return result


class InterpFormContentDict(SvFormContentDict):
    """This class is present for backwards compatibility only."""
    def __getitem__(self, key):
        v = SvFormContentDict.__getitem__(self, key)
        if v[0] in '0123456789+-.':
            try: return int(v)
            except ValueError:
                try: return float(v)
                except ValueError: pass
        return v.strip()
    def values(self):
        result = []
        for key in self.keys():
            try:
                result.append(self[key])
            except IndexError:
                result.append(self.dict[key])
        return result
    def items(self):
        result = []
        for key in self.keys():
            try:
                result.append((key, self[key]))
            except IndexError:
                result.append((key, self.dict[key]))
        return result


class FormContent(FormContentDict):
    """This class is present for backwards compatibility only."""
    def values(self, key):
        if key in self.dict :return self.dict[key]
        else: return None
    def indexed_value(self, key, location):
        if key in self.dict:
            if len(self.dict[key]) > location:
                return self.dict[key][location]
            else: return None
        else: return None
    def value(self, key):
        if key in self.dict: return self.dict[key][0]
        else: return None
    def length(self, key):
        return len(self.dict[key])
    def stripped(self, key):
        if key in self.dict: return self.dict[key][0].strip()
        else: return None
    def pars(self):
        return self.dict


# Test/debug code
# ===============

def test(environ=os.environ):
    """Robust test CGI script, usable as main program.

    Write minimal HTTP headers and dump all information provided to
    the script in HTML form.

    """
    print "Content-type: text/html"
    print
    sys.stderr = sys.stdout
    try:
        form = FieldStorage()   # Replace with other classes to test those
        print_directory()
        print_arguments()
        print_form(form)
        print_environ(environ)
        print_environ_usage()
        def f():
            exec "testing print_exception() -- <I>italics?</I>"
        def g(f=f):
            f()
        print "<H3>What follows is a test, not an actual exception:</H3>"
        g()
    except:
        print_exception()

    print "<H1>Second try with a small maxlen...</H1>"

    global maxlen
    maxlen = 50
    try:
        form = FieldStorage()   # Replace with other classes to test those
        print_directory()
        print_arguments()
        print_form(form)
        print_environ(environ)
    except:
        print_exception()

def print_exception(type=None, value=None, tb=None, limit=None):
    if type is None:
        type, value, tb = sys.exc_info()
    import traceback
    print
    print "<H3>Traceback (most recent call last):</H3>"
    list = traceback.format_tb(tb, limit) + \
           traceback.format_exception_only(type, value)
    print "<PRE>%s<B>%s</B></PRE>" % (
        escape("".join(list[:-1])),
        escape(list[-1]),
        )
    del tb

def print_environ(environ=os.environ):
    """Dump the shell environment as HTML."""
    keys = environ.keys()
    keys.sort()
    print
    print "<H3>Shell Environment:</H3>"
    print "<DL>"
    for key in keys:
        print "<DT>", escape(key), "<DD>", escape(environ[key])
    print "</DL>"
    print

def print_form(form):
    """Dump the contents of a form as HTML."""
    keys = form.keys()
    keys.sort()
    print
    print "<H3>Form Contents:</H3>"
    if not keys:
        print "<P>No form fields."
    print "<DL>"
    for key in keys:
        print "<DT>" + escape(key) + ":",
        value = form[key]
        print "<i>" + escape(repr(type(value))) + "</i>"
        print "<DD>" + escape(repr(value))
    print "</DL>"
    print

def print_directory():
    """Dump the current directory as HTML."""
    print
    print "<H3>Current Working Directory:</H3>"
    try:
        pwd = os.getcwd()
    except os.error, msg:
        print "os.error:", escape(str(msg))
    else:
        print escape(pwd)
    print

def print_arguments():
    print
    print "<H3>Command Line Arguments:</H3>"
    print
    print sys.argv
    print

def print_environ_usage():
    """Dump a list of environment variables used by CGI as HTML."""
    print """
<H3>These environment variables could have been set:</H3>
<UL>
<LI>AUTH_TYPE
<LI>CONTENT_LENGTH
<LI>CONTENT_TYPE
<LI>DATE_GMT
<LI>DATE_LOCAL
<LI>DOCUMENT_NAME
<LI>DOCUMENT_ROOT
<LI>DOCUMENT_URI
<LI>GATEWAY_INTERFACE
<LI>LAST_MODIFIED
<LI>PATH
<LI>PATH_INFO
<LI>PATH_TRANSLATED
<LI>QUERY_STRING
<LI>REMOTE_ADDR
<LI>REMOTE_HOST
<LI>REMOTE_IDENT
<LI>REMOTE_USER
<LI>REQUEST_METHOD
<LI>SCRIPT_NAME
<LI>SERVER_NAME
<LI>SERVER_PORT
<LI>SERVER_PROTOCOL
<LI>SERVER_ROOT
<LI>SERVER_SOFTWARE
</UL>
In addition, HTTP headers sent by the server may be passed in the
environment as well.  Here are some common variable names:
<UL>
<LI>HTTP_ACCEPT
<LI>HTTP_CONNECTION
<LI>HTTP_HOST
<LI>HTTP_PRAGMA
<LI>HTTP_REFERER
<LI>HTTP_USER_AGENT
</UL>
"""


# Utilities
# =========

def escape(s, quote=None):
    '''Replace special characters "&", "<" and ">" to HTML-safe sequences.
    If the optional flag quote is true, the quotation mark character (")
    is also translated.'''
    s = s.replace("&", "&amp;") # Must be done first!
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s

def valid_boundary(s, _vb_pattern="^[ -~]{0,200}[!-~]$"):
    import re
    return re.match(_vb_pattern, s)

# Invoke mainline
# ===============

# Call test() when this file is run as a script (not imported as a module)
if __name__ == '__main__':
    test()
