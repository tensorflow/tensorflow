# -*- coding: utf-8 -*-
"""
    werkzeug.formparser
    ~~~~~~~~~~~~~~~~~~~

    This module implements the form parsing.  It supports url-encoded forms
    as well as non-nested multipart uploads.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import codecs
from io import BytesIO
from tempfile import TemporaryFile
from itertools import chain, repeat, tee
from functools import update_wrapper

from werkzeug._compat import to_native, text_type
from werkzeug.urls import url_decode_stream
from werkzeug.wsgi import make_line_iter, \
    get_input_stream, get_content_length
from werkzeug.datastructures import Headers, FileStorage, MultiDict
from werkzeug.http import parse_options_header


#: an iterator that yields empty strings
_empty_string_iter = repeat('')

#: a regular expression for multipart boundaries
_multipart_boundary_re = re.compile('^[ -~]{0,200}[!-~]$')

#: supported http encodings that are also available in python we support
#: for multipart messages.
_supported_multipart_encodings = frozenset(['base64', 'quoted-printable'])


def default_stream_factory(total_content_length, filename, content_type,
                           content_length=None):
    """The stream factory that is used per default."""
    if total_content_length > 1024 * 500:
        return TemporaryFile('wb+')
    return BytesIO()


def parse_form_data(environ, stream_factory=None, charset='utf-8',
                    errors='replace', max_form_memory_size=None,
                    max_content_length=None, cls=None,
                    silent=True):
    """Parse the form data in the environ and return it as tuple in the form
    ``(stream, form, files)``.  You should only call this method if the
    transport method is `POST`, `PUT`, or `PATCH`.

    If the mimetype of the data transmitted is `multipart/form-data` the
    files multidict will be filled with `FileStorage` objects.  If the
    mimetype is unknown the input stream is wrapped and returned as first
    argument, else the stream is empty.

    This is a shortcut for the common usage of :class:`FormDataParser`.

    Have a look at :ref:`dealing-with-request-data` for more details.

    .. versionadded:: 0.5
       The `max_form_memory_size`, `max_content_length` and
       `cls` parameters were added.

    .. versionadded:: 0.5.1
       The optional `silent` flag was added.

    :param environ: the WSGI environment to be used for parsing.
    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`~BaseResponse._get_file_stream`.
    :param charset: The character set for URL and url encoded form data.
    :param errors: The encoding error behavior.
    :param max_form_memory_size: the maximum number of bytes to be accepted for
                           in-memory stored form data.  If the data
                           exceeds the value specified an
                           :exc:`~exceptions.RequestEntityTooLarge`
                           exception is raised.
    :param max_content_length: If this is provided and the transmitted data
                               is longer than this value an
                               :exc:`~exceptions.RequestEntityTooLarge`
                               exception is raised.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    :return: A tuple in the form ``(stream, form, files)``.
    """
    return FormDataParser(stream_factory, charset, errors,
                          max_form_memory_size, max_content_length,
                          cls, silent).parse_from_environ(environ)


def exhaust_stream(f):
    """Helper decorator for methods that exhausts the stream on return."""

    def wrapper(self, stream, *args, **kwargs):
        try:
            return f(self, stream, *args, **kwargs)
        finally:
            exhaust = getattr(stream, 'exhaust', None)
            if exhaust is not None:
                exhaust()
            else:
                while 1:
                    chunk = stream.read(1024 * 64)
                    if not chunk:
                        break
    return update_wrapper(wrapper, f)


class FormDataParser(object):

    """This class implements parsing of form data for Werkzeug.  By itself
    it can parse multipart and url encoded form data.  It can be subclassed
    and extended but for most mimetypes it is a better idea to use the
    untouched stream and expose it as separate attributes on a request
    object.

    .. versionadded:: 0.8

    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`~BaseResponse._get_file_stream`.
    :param charset: The character set for URL and url encoded form data.
    :param errors: The encoding error behavior.
    :param max_form_memory_size: the maximum number of bytes to be accepted for
                           in-memory stored form data.  If the data
                           exceeds the value specified an
                           :exc:`~exceptions.RequestEntityTooLarge`
                           exception is raised.
    :param max_content_length: If this is provided and the transmitted data
                               is longer than this value an
                               :exc:`~exceptions.RequestEntityTooLarge`
                               exception is raised.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    """

    def __init__(self, stream_factory=None, charset='utf-8',
                 errors='replace', max_form_memory_size=None,
                 max_content_length=None, cls=None,
                 silent=True):
        if stream_factory is None:
            stream_factory = default_stream_factory
        self.stream_factory = stream_factory
        self.charset = charset
        self.errors = errors
        self.max_form_memory_size = max_form_memory_size
        self.max_content_length = max_content_length
        if cls is None:
            cls = MultiDict
        self.cls = cls
        self.silent = silent

    def get_parse_func(self, mimetype, options):
        return self.parse_functions.get(mimetype)

    def parse_from_environ(self, environ):
        """Parses the information from the environment as form data.

        :param environ: the WSGI environment to be used for parsing.
        :return: A tuple in the form ``(stream, form, files)``.
        """
        content_type = environ.get('CONTENT_TYPE', '')
        content_length = get_content_length(environ)
        mimetype, options = parse_options_header(content_type)
        return self.parse(get_input_stream(environ), mimetype,
                          content_length, options)

    def parse(self, stream, mimetype, content_length, options=None):
        """Parses the information from the given stream, mimetype,
        content length and mimetype parameters.

        :param stream: an input stream
        :param mimetype: the mimetype of the data
        :param content_length: the content length of the incoming data
        :param options: optional mimetype parameters (used for
                        the multipart boundary for instance)
        :return: A tuple in the form ``(stream, form, files)``.
        """
        if self.max_content_length is not None and \
           content_length is not None and \
           content_length > self.max_content_length:
            raise exceptions.RequestEntityTooLarge()
        if options is None:
            options = {}

        parse_func = self.get_parse_func(mimetype, options)
        if parse_func is not None:
            try:
                return parse_func(self, stream, mimetype,
                                  content_length, options)
            except ValueError:
                if not self.silent:
                    raise

        return stream, self.cls(), self.cls()

    @exhaust_stream
    def _parse_multipart(self, stream, mimetype, content_length, options):
        parser = MultiPartParser(self.stream_factory, self.charset, self.errors,
                                 max_form_memory_size=self.max_form_memory_size,
                                 cls=self.cls)
        boundary = options.get('boundary')
        if boundary is None:
            raise ValueError('Missing boundary')
        if isinstance(boundary, text_type):
            boundary = boundary.encode('ascii')
        form, files = parser.parse(stream, boundary, content_length)
        return stream, form, files

    @exhaust_stream
    def _parse_urlencoded(self, stream, mimetype, content_length, options):
        if self.max_form_memory_size is not None and \
           content_length is not None and \
           content_length > self.max_form_memory_size:
            raise exceptions.RequestEntityTooLarge()
        form = url_decode_stream(stream, self.charset,
                                 errors=self.errors, cls=self.cls)
        return stream, form, self.cls()

    #: mapping of mimetypes to parsing functions
    parse_functions = {
        'multipart/form-data':                  _parse_multipart,
        'application/x-www-form-urlencoded':    _parse_urlencoded,
        'application/x-url-encoded':            _parse_urlencoded
    }


def is_valid_multipart_boundary(boundary):
    """Checks if the string given is a valid multipart boundary."""
    return _multipart_boundary_re.match(boundary) is not None


def _line_parse(line):
    """Removes line ending characters and returns a tuple (`stripped_line`,
    `is_terminated`).
    """
    if line[-2:] in ['\r\n', b'\r\n']:
        return line[:-2], True
    elif line[-1:] in ['\r', '\n', b'\r', b'\n']:
        return line[:-1], True
    return line, False


def parse_multipart_headers(iterable):
    """Parses multipart headers from an iterable that yields lines (including
    the trailing newline symbol).  The iterable has to be newline terminated.

    The iterable will stop at the line where the headers ended so it can be
    further consumed.

    :param iterable: iterable of strings that are newline terminated
    """
    result = []
    for line in iterable:
        line = to_native(line)
        line, line_terminated = _line_parse(line)
        if not line_terminated:
            raise ValueError('unexpected end of line in multipart header')
        if not line:
            break
        elif line[0] in ' \t' and result:
            key, value = result[-1]
            result[-1] = (key, value + '\n ' + line[1:])
        else:
            parts = line.split(':', 1)
            if len(parts) == 2:
                result.append((parts[0].strip(), parts[1].strip()))

    # we link the list to the headers, no need to create a copy, the
    # list was not shared anyways.
    return Headers(result)


_begin_form = 'begin_form'
_begin_file = 'begin_file'
_cont = 'cont'
_end = 'end'


class MultiPartParser(object):

    def __init__(self, stream_factory=None, charset='utf-8', errors='replace',
                 max_form_memory_size=None, cls=None, buffer_size=64 * 1024):
        self.stream_factory = stream_factory
        self.charset = charset
        self.errors = errors
        self.max_form_memory_size = max_form_memory_size
        if stream_factory is None:
            stream_factory = default_stream_factory
        if cls is None:
            cls = MultiDict
        self.cls = cls

        # make sure the buffer size is divisible by four so that we can base64
        # decode chunk by chunk
        assert buffer_size % 4 == 0, 'buffer size has to be divisible by 4'
        # also the buffer size has to be at least 1024 bytes long or long headers
        # will freak out the system
        assert buffer_size >= 1024, 'buffer size has to be at least 1KB'

        self.buffer_size = buffer_size

    def _fix_ie_filename(self, filename):
        """Internet Explorer 6 transmits the full file name if a file is
        uploaded.  This function strips the full path if it thinks the
        filename is Windows-like absolute.
        """
        if filename[1:3] == ':\\' or filename[:2] == '\\\\':
            return filename.split('\\')[-1]
        return filename

    def _find_terminator(self, iterator):
        """The terminator might have some additional newlines before it.
        There is at least one application that sends additional newlines
        before headers (the python setuptools package).
        """
        for line in iterator:
            if not line:
                break
            line = line.strip()
            if line:
                return line
        return b''

    def fail(self, message):
        raise ValueError(message)

    def get_part_encoding(self, headers):
        transfer_encoding = headers.get('content-transfer-encoding')
        if transfer_encoding is not None and \
           transfer_encoding in _supported_multipart_encodings:
            return transfer_encoding

    def get_part_charset(self, headers):
        # Figure out input charset for current part
        content_type = headers.get('content-type')
        if content_type:
            mimetype, ct_params = parse_options_header(content_type)
            return ct_params.get('charset', self.charset)
        return self.charset

    def start_file_streaming(self, filename, headers, total_content_length):
        if isinstance(filename, bytes):
            filename = filename.decode(self.charset, self.errors)
        filename = self._fix_ie_filename(filename)
        content_type = headers.get('content-type')
        try:
            content_length = int(headers['content-length'])
        except (KeyError, ValueError):
            content_length = 0
        container = self.stream_factory(total_content_length, content_type,
                                        filename, content_length)
        return filename, container

    def in_memory_threshold_reached(self, bytes):
        raise exceptions.RequestEntityTooLarge()

    def validate_boundary(self, boundary):
        if not boundary:
            self.fail('Missing boundary')
        if not is_valid_multipart_boundary(boundary):
            self.fail('Invalid boundary: %s' % boundary)
        if len(boundary) > self.buffer_size:  # pragma: no cover
            # this should never happen because we check for a minimum size
            # of 1024 and boundaries may not be longer than 200.  The only
            # situation when this happens is for non debug builds where
            # the assert is skipped.
            self.fail('Boundary longer than buffer size')

    def parse_lines(self, file, boundary, content_length, cap_at_buffer=True):
        """Generate parts of
        ``('begin_form', (headers, name))``
        ``('begin_file', (headers, name, filename))``
        ``('cont', bytestring)``
        ``('end', None)``

        Always obeys the grammar
        parts = ( begin_form cont* end |
                  begin_file cont* end )*
        """
        next_part = b'--' + boundary
        last_part = next_part + b'--'

        iterator = chain(make_line_iter(file, limit=content_length,
                                        buffer_size=self.buffer_size,
                                        cap_at_buffer=cap_at_buffer),
                         _empty_string_iter)

        terminator = self._find_terminator(iterator)

        if terminator == last_part:
            return
        elif terminator != next_part:
            self.fail('Expected boundary at start of multipart data')

        while terminator != last_part:
            headers = parse_multipart_headers(iterator)

            disposition = headers.get('content-disposition')
            if disposition is None:
                self.fail('Missing Content-Disposition header')
            disposition, extra = parse_options_header(disposition)
            transfer_encoding = self.get_part_encoding(headers)
            name = extra.get('name')
            filename = extra.get('filename')

            # if no content type is given we stream into memory.  A list is
            # used as a temporary container.
            if filename is None:
                yield _begin_form, (headers, name)

            # otherwise we parse the rest of the headers and ask the stream
            # factory for something we can write in.
            else:
                yield _begin_file, (headers, name, filename)

            buf = b''
            for line in iterator:
                if not line:
                    self.fail('unexpected end of stream')

                if line[:2] == b'--':
                    terminator = line.rstrip()
                    if terminator in (next_part, last_part):
                        break

                if transfer_encoding is not None:
                    if transfer_encoding == 'base64':
                        transfer_encoding = 'base64_codec'
                    try:
                        line = codecs.decode(line, transfer_encoding)
                    except Exception:
                        self.fail('could not decode transfer encoded chunk')

                # we have something in the buffer from the last iteration.
                # this is usually a newline delimiter.
                if buf:
                    yield _cont, buf
                    buf = b''

                # If the line ends with windows CRLF we write everything except
                # the last two bytes.  In all other cases however we write
                # everything except the last byte.  If it was a newline, that's
                # fine, otherwise it does not matter because we will write it
                # the next iteration.  this ensures we do not write the
                # final newline into the stream.  That way we do not have to
                # truncate the stream.  However we do have to make sure that
                # if something else than a newline is in there we write it
                # out.
                if line[-2:] == b'\r\n':
                    buf = b'\r\n'
                    cutoff = -2
                else:
                    buf = line[-1:]
                    cutoff = -1
                yield _cont, line[:cutoff]

            else:  # pragma: no cover
                raise ValueError('unexpected end of part')

            # if we have a leftover in the buffer that is not a newline
            # character we have to flush it, otherwise we will chop of
            # certain values.
            if buf not in (b'', b'\r', b'\n', b'\r\n'):
                yield _cont, buf

            yield _end, None

    def parse_parts(self, file, boundary, content_length):
        """Generate ``('file', (name, val))`` and
        ``('form', (name, val))`` parts.
        """
        in_memory = 0

        for ellt, ell in self.parse_lines(file, boundary, content_length):
            if ellt == _begin_file:
                headers, name, filename = ell
                is_file = True
                guard_memory = False
                filename, container = self.start_file_streaming(
                    filename, headers, content_length)
                _write = container.write

            elif ellt == _begin_form:
                headers, name = ell
                is_file = False
                container = []
                _write = container.append
                guard_memory = self.max_form_memory_size is not None

            elif ellt == _cont:
                _write(ell)
                # if we write into memory and there is a memory size limit we
                # count the number of bytes in memory and raise an exception if
                # there is too much data in memory.
                if guard_memory:
                    in_memory += len(ell)
                    if in_memory > self.max_form_memory_size:
                        self.in_memory_threshold_reached(in_memory)

            elif ellt == _end:
                if is_file:
                    container.seek(0)
                    yield ('file',
                           (name, FileStorage(container, filename, name,
                                              headers=headers)))
                else:
                    part_charset = self.get_part_charset(headers)
                    yield ('form',
                           (name, b''.join(container).decode(
                               part_charset, self.errors)))

    def parse(self, file, boundary, content_length):
        formstream, filestream = tee(
            self.parse_parts(file, boundary, content_length), 2)
        form = (p[1] for p in formstream if p[0] == 'form')
        files = (p[1] for p in filestream if p[0] == 'file')
        return self.cls(form), self.cls(files)


from werkzeug import exceptions
