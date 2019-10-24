"""Guess the MIME type of a file.

This module defines two useful functions:

guess_type(url, strict=1) -- guess the MIME type and encoding of a URL.

guess_extension(type, strict=1) -- guess the extension for a given MIME type.

It also contains the following, for tuning the behavior:

Data:

knownfiles -- list of files to parse
inited -- flag set when init() has been called
suffix_map -- dictionary mapping suffixes to suffixes
encodings_map -- dictionary mapping suffixes to encodings
types_map -- dictionary mapping suffixes to types

Functions:

init([files]) -- parse a list of files, default knownfiles
read_mime_types(file) -- parse one file, return a dictionary or None
"""

import os
import posixpath
import urllib

__all__ = [
    "guess_type","guess_extension","guess_all_extensions",
    "add_type","read_mime_types","init"
]

knownfiles = [
    "/etc/mime.types",
    "/etc/httpd/mime.types",                    # Mac OS X
    "/etc/httpd/conf/mime.types",               # Apache
    "/etc/apache/mime.types",                   # Apache 1
    "/etc/apache2/mime.types",                  # Apache 2
    "/usr/local/etc/httpd/conf/mime.types",
    "/usr/local/lib/netscape/mime.types",
    "/usr/local/etc/httpd/conf/mime.types",     # Apache 1.2
    "/usr/local/etc/mime.types",                # Apache 1.3
    ]

inited = False


class MimeTypes:
    """MIME-types datastore.

    This datastore can handle information from mime.types-style files
    and supports basic determination of MIME type from a filename or
    URL, and can guess a reasonable extension given a MIME type.
    """

    def __init__(self, filenames=(), strict=True):
        if not inited:
            init()
        self.encodings_map = encodings_map.copy()
        self.suffix_map = suffix_map.copy()
        self.types_map = ({}, {}) # dict for (non-strict, strict)
        self.types_map_inv = ({}, {})
        for (ext, type) in types_map.items():
            self.add_type(type, ext, True)
        for (ext, type) in common_types.items():
            self.add_type(type, ext, False)
        for name in filenames:
            self.read(name, strict)

    def add_type(self, type, ext, strict=True):
        """Add a mapping between a type and an extension.

        When the extension is already known, the new
        type will replace the old one. When the type
        is already known the extension will be added
        to the list of known extensions.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
        self.types_map[strict][ext] = type
        exts = self.types_map_inv[strict].setdefault(type, [])
        if ext not in exts:
            exts.append(ext)

    def guess_type(self, url, strict=True):
        """Guess the type of a file based on its URL.

        Return value is a tuple (type, encoding) where type is None if
        the type can't be guessed (no or unknown suffix) or a string
        of the form type/subtype, usable for a MIME Content-type
        header; and encoding is None for no encoding or the name of
        the program used to encode (e.g. compress or gzip).  The
        mappings are table driven.  Encoding suffixes are case
        sensitive; type suffixes are first tried case sensitive, then
        case insensitive.

        The suffixes .tgz, .taz and .tz (case sensitive!) are all
        mapped to '.tar.gz'.  (This is table-driven too, using the
        dictionary suffix_map.)

        Optional `strict' argument when False adds a bunch of commonly found,
        but non-standard types.
        """
        scheme, url = urllib.splittype(url)
        if scheme == 'data':
            # syntax of data URLs:
            # dataurl   := "data:" [ mediatype ] [ ";base64" ] "," data
            # mediatype := [ type "/" subtype ] *( ";" parameter )
            # data      := *urlchar
            # parameter := attribute "=" value
            # type/subtype defaults to "text/plain"
            comma = url.find(',')
            if comma < 0:
                # bad data URL
                return None, None
            semi = url.find(';', 0, comma)
            if semi >= 0:
                type = url[:semi]
            else:
                type = url[:comma]
            if '=' in type or '/' not in type:
                type = 'text/plain'
            return type, None           # never compressed, so encoding is None
        base, ext = posixpath.splitext(url)
        while ext in self.suffix_map:
            base, ext = posixpath.splitext(base + self.suffix_map[ext])
        if ext in self.encodings_map:
            encoding = self.encodings_map[ext]
            base, ext = posixpath.splitext(base)
        else:
            encoding = None
        types_map = self.types_map[True]
        if ext in types_map:
            return types_map[ext], encoding
        elif ext.lower() in types_map:
            return types_map[ext.lower()], encoding
        elif strict:
            return None, encoding
        types_map = self.types_map[False]
        if ext in types_map:
            return types_map[ext], encoding
        elif ext.lower() in types_map:
            return types_map[ext.lower()], encoding
        else:
            return None, encoding

    def guess_all_extensions(self, type, strict=True):
        """Guess the extensions for a file based on its MIME type.

        Return value is a list of strings giving the possible filename
        extensions, including the leading dot ('.').  The extension is not
        guaranteed to have been associated with any particular data stream,
        but would be mapped to the MIME type `type' by guess_type().

        Optional `strict' argument when false adds a bunch of commonly found,
        but non-standard types.
        """
        type = type.lower()
        extensions = self.types_map_inv[True].get(type, [])
        if not strict:
            for ext in self.types_map_inv[False].get(type, []):
                if ext not in extensions:
                    extensions.append(ext)
        return extensions

    def guess_extension(self, type, strict=True):
        """Guess the extension for a file based on its MIME type.

        Return value is a string giving a filename extension,
        including the leading dot ('.').  The extension is not
        guaranteed to have been associated with any particular data
        stream, but would be mapped to the MIME type `type' by
        guess_type().  If no extension can be guessed for `type', None
        is returned.

        Optional `strict' argument when false adds a bunch of commonly found,
        but non-standard types.
        """
        extensions = self.guess_all_extensions(type, strict)
        if not extensions:
            return None
        return extensions[0]

    def read(self, filename, strict=True):
        """
        Read a single mime.types-format file, specified by pathname.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
        fp = open(filename)
        self.readfp(fp, strict)
        fp.close()

    def readfp(self, fp, strict=True):
        """
        Read a single mime.types-format file.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
        while 1:
            line = fp.readline()
            if not line:
                break
            words = line.split()
            for i in range(len(words)):
                if words[i][0] == '#':
                    del words[i:]
                    break
            if not words:
                continue
            type, suffixes = words[0], words[1:]
            for suff in suffixes:
                self.add_type(type, '.' + suff, strict)

def guess_type(url, strict=True):
    """Guess the type of a file based on its URL.

    Return value is a tuple (type, encoding) where type is None if the
    type can't be guessed (no or unknown suffix) or a string of the
    form type/subtype, usable for a MIME Content-type header; and
    encoding is None for no encoding or the name of the program used
    to encode (e.g. compress or gzip).  The mappings are table
    driven.  Encoding suffixes are case sensitive; type suffixes are
    first tried case sensitive, then case insensitive.

    The suffixes .tgz, .taz and .tz (case sensitive!) are all mapped
    to ".tar.gz".  (This is table-driven too, using the dictionary
    suffix_map).

    Optional `strict' argument when false adds a bunch of commonly found, but
    non-standard types.
    """
    init()
    return guess_type(url, strict)


def guess_all_extensions(type, strict=True):
    """Guess the extensions for a file based on its MIME type.

    Return value is a list of strings giving the possible filename
    extensions, including the leading dot ('.').  The extension is not
    guaranteed to have been associated with any particular data
    stream, but would be mapped to the MIME type `type' by
    guess_type().  If no extension can be guessed for `type', None
    is returned.

    Optional `strict' argument when false adds a bunch of commonly found,
    but non-standard types.
    """
    init()
    return guess_all_extensions(type, strict)

def guess_extension(type, strict=True):
    """Guess the extension for a file based on its MIME type.

    Return value is a string giving a filename extension, including the
    leading dot ('.').  The extension is not guaranteed to have been
    associated with any particular data stream, but would be mapped to the
    MIME type `type' by guess_type().  If no extension can be guessed for
    `type', None is returned.

    Optional `strict' argument when false adds a bunch of commonly found,
    but non-standard types.
    """
    init()
    return guess_extension(type, strict)

def add_type(type, ext, strict=True):
    """Add a mapping between a type and an extension.

    When the extension is already known, the new
    type will replace the old one. When the type
    is already known the extension will be added
    to the list of known extensions.

    If strict is true, information will be added to
    list of standard types, else to the list of non-standard
    types.
    """
    init()
    return add_type(type, ext, strict)


def init(files=None):
    global guess_all_extensions, guess_extension, guess_type
    global suffix_map, types_map, encodings_map, common_types
    global add_type, inited
    inited = True
    db = MimeTypes()
    if files is None:
        files = knownfiles
    for file in files:
        if os.path.isfile(file):
            db.readfp(open(file))
    encodings_map = db.encodings_map
    suffix_map = db.suffix_map
    types_map = db.types_map[True]
    guess_all_extensions = db.guess_all_extensions
    guess_extension = db.guess_extension
    guess_type = db.guess_type
    add_type = db.add_type
    common_types = db.types_map[False]


def read_mime_types(file):
    try:
        f = open(file)
    except IOError:
        return None
    db = MimeTypes()
    db.readfp(f, True)
    return db.types_map[True]


def _default_mime_types():
    global suffix_map
    global encodings_map
    global types_map
    global common_types

    suffix_map = {
        '.tgz': '.tar.gz',
        '.taz': '.tar.gz',
        '.tz': '.tar.gz',
        }

    encodings_map = {
        '.gz': 'gzip',
        '.Z': 'compress',
        }

    # Before adding new types, make sure they are either registered with IANA,
    # at http://www.isi.edu/in-notes/iana/assignments/media-types
    # or extensions, i.e. using the x- prefix

    # If you add to these, please keep them sorted!
    types_map = {
        '.a'      : 'application/octet-stream',
        '.ai'     : 'application/postscript',
        '.aif'    : 'audio/x-aiff',
        '.aifc'   : 'audio/x-aiff',
        '.aiff'   : 'audio/x-aiff',
        '.au'     : 'audio/basic',
        '.avi'    : 'video/x-msvideo',
        '.bat'    : 'text/plain',
        '.bcpio'  : 'application/x-bcpio',
        '.bin'    : 'application/octet-stream',
        '.bmp'    : 'image/x-ms-bmp',
        '.c'      : 'text/plain',
        # Duplicates :(
        '.cdf'    : 'application/x-cdf',
        '.cdf'    : 'application/x-netcdf',
        '.cpio'   : 'application/x-cpio',
        '.csh'    : 'application/x-csh',
        '.css'    : 'text/css',
        '.dll'    : 'application/octet-stream',
        '.doc'    : 'application/msword',
        '.dot'    : 'application/msword',
        '.dvi'    : 'application/x-dvi',
        '.eml'    : 'message/rfc822',
        '.eps'    : 'application/postscript',
        '.etx'    : 'text/x-setext',
        '.exe'    : 'application/octet-stream',
        '.gif'    : 'image/gif',
        '.gtar'   : 'application/x-gtar',
        '.h'      : 'text/plain',
        '.hdf'    : 'application/x-hdf',
        '.htm'    : 'text/html',
        '.html'   : 'text/html',
        '.ief'    : 'image/ief',
        '.jpe'    : 'image/jpeg',
        '.jpeg'   : 'image/jpeg',
        '.jpg'    : 'image/jpeg',
        '.js'     : 'application/x-javascript',
        '.ksh'    : 'text/plain',
        '.latex'  : 'application/x-latex',
        '.m1v'    : 'video/mpeg',
        '.man'    : 'application/x-troff-man',
        '.me'     : 'application/x-troff-me',
        '.mht'    : 'message/rfc822',
        '.mhtml'  : 'message/rfc822',
        '.mif'    : 'application/x-mif',
        '.mov'    : 'video/quicktime',
        '.movie'  : 'video/x-sgi-movie',
        '.mp2'    : 'audio/mpeg',
        '.mp3'    : 'audio/mpeg',
        '.mpa'    : 'video/mpeg',
        '.mpe'    : 'video/mpeg',
        '.mpeg'   : 'video/mpeg',
        '.mpg'    : 'video/mpeg',
        '.ms'     : 'application/x-troff-ms',
        '.nc'     : 'application/x-netcdf',
        '.nws'    : 'message/rfc822',
        '.o'      : 'application/octet-stream',
        '.obj'    : 'application/octet-stream',
        '.oda'    : 'application/oda',
        '.p12'    : 'application/x-pkcs12',
        '.p7c'    : 'application/pkcs7-mime',
        '.pbm'    : 'image/x-portable-bitmap',
        '.pdf'    : 'application/pdf',
        '.pfx'    : 'application/x-pkcs12',
        '.pgm'    : 'image/x-portable-graymap',
        '.pl'     : 'text/plain',
        '.png'    : 'image/png',
        '.pnm'    : 'image/x-portable-anymap',
        '.pot'    : 'application/vnd.ms-powerpoint',
        '.ppa'    : 'application/vnd.ms-powerpoint',
        '.ppm'    : 'image/x-portable-pixmap',
        '.pps'    : 'application/vnd.ms-powerpoint',
        '.ppt'    : 'application/vnd.ms-powerpoint',
        '.ps'     : 'application/postscript',
        '.pwz'    : 'application/vnd.ms-powerpoint',
        '.py'     : 'text/x-python',
        '.pyc'    : 'application/x-python-code',
        '.pyo'    : 'application/x-python-code',
        '.qt'     : 'video/quicktime',
        '.ra'     : 'audio/x-pn-realaudio',
        '.ram'    : 'application/x-pn-realaudio',
        '.ras'    : 'image/x-cmu-raster',
        '.rdf'    : 'application/xml',
        '.rgb'    : 'image/x-rgb',
        '.roff'   : 'application/x-troff',
        '.rtx'    : 'text/richtext',
        '.sgm'    : 'text/x-sgml',
        '.sgml'   : 'text/x-sgml',
        '.sh'     : 'application/x-sh',
        '.shar'   : 'application/x-shar',
        '.snd'    : 'audio/basic',
        '.so'     : 'application/octet-stream',
        '.src'    : 'application/x-wais-source',
        '.sv4cpio': 'application/x-sv4cpio',
        '.sv4crc' : 'application/x-sv4crc',
        '.swf'    : 'application/x-shockwave-flash',
        '.t'      : 'application/x-troff',
        '.tar'    : 'application/x-tar',
        '.tcl'    : 'application/x-tcl',
        '.tex'    : 'application/x-tex',
        '.texi'   : 'application/x-texinfo',
        '.texinfo': 'application/x-texinfo',
        '.tif'    : 'image/tiff',
        '.tiff'   : 'image/tiff',
        '.tr'     : 'application/x-troff',
        '.tsv'    : 'text/tab-separated-values',
        '.txt'    : 'text/plain',
        '.ustar'  : 'application/x-ustar',
        '.vcf'    : 'text/x-vcard',
        '.wav'    : 'audio/x-wav',
        '.wiz'    : 'application/msword',
        '.wsdl'   : 'application/xml',
        '.xbm'    : 'image/x-xbitmap',
        '.xlb'    : 'application/vnd.ms-excel',
        # Duplicates :(
        '.xls'    : 'application/excel',
        '.xls'    : 'application/vnd.ms-excel',
        '.xml'    : 'text/xml',
        '.xpdl'   : 'application/xml',
        '.xpm'    : 'image/x-xpixmap',
        '.xsl'    : 'application/xml',
        '.xwd'    : 'image/x-xwindowdump',
        '.zip'    : 'application/zip',
        }

    # These are non-standard types, commonly found in the wild.  They will
    # only match if strict=0 flag is given to the API methods.

    # Please sort these too
    common_types = {
        '.jpg' : 'image/jpg',
        '.mid' : 'audio/midi',
        '.midi': 'audio/midi',
        '.pct' : 'image/pict',
        '.pic' : 'image/pict',
        '.pict': 'image/pict',
        '.rtf' : 'application/rtf',
        '.xul' : 'text/xul'
        }


_default_mime_types()


if __name__ == '__main__':
    import sys
    import getopt

    USAGE = """\
Usage: mimetypes.py [options] type

Options:
    --help / -h       -- print this message and exit
    --lenient / -l    -- additionally search of some common, but non-standard
                         types.
    --extension / -e  -- guess extension instead of type

More than one type argument may be given.
"""

    def usage(code, msg=''):
        print USAGE
        if msg: print msg
        sys.exit(code)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hle',
                                   ['help', 'lenient', 'extension'])
    except getopt.error, msg:
        usage(1, msg)

    strict = 1
    extension = 0
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage(0)
        elif opt in ('-l', '--lenient'):
            strict = 0
        elif opt in ('-e', '--extension'):
            extension = 1
    for gtype in args:
        if extension:
            guess = guess_extension(gtype, strict)
            if not guess: print "I don't know anything about type", gtype
            else: print guess
        else:
            guess, encoding = guess_type(gtype, strict)
            if not guess: print "I don't know anything about type", gtype
            else: print 'type:', guess, 'encoding:', encoding
