# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.wrappers
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Extra wrappers or mixins contributed by the community.  These wrappers can
    be mixed in into request objects to add extra functionality.

    Example::

        from werkzeug.wrappers import Request as RequestBase
        from werkzeug.contrib.wrappers import JSONRequestMixin

        class Request(RequestBase, JSONRequestMixin):
            pass

    Afterwards this request object provides the extra functionality of the
    :class:`JSONRequestMixin`.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import codecs
try:
    from simplejson import loads
except ImportError:
    from json import loads

from werkzeug.exceptions import BadRequest
from werkzeug.utils import cached_property
from werkzeug.http import dump_options_header, parse_options_header
from werkzeug._compat import wsgi_decoding_dance


def is_known_charset(charset):
    """Checks if the given charset is known to Python."""
    try:
        codecs.lookup(charset)
    except LookupError:
        return False
    return True


class JSONRequestMixin(object):

    """Add json method to a request object.  This will parse the input data
    through simplejson if possible.

    :exc:`~werkzeug.exceptions.BadRequest` will be raised if the content-type
    is not json or if the data itself cannot be parsed as json.
    """

    @cached_property
    def json(self):
        """Get the result of simplejson.loads if possible."""
        if 'json' not in self.environ.get('CONTENT_TYPE', ''):
            raise BadRequest('Not a JSON request')
        try:
            return loads(self.data.decode(self.charset, self.encoding_errors))
        except Exception:
            raise BadRequest('Unable to read JSON request')


class ProtobufRequestMixin(object):

    """Add protobuf parsing method to a request object.  This will parse the
    input data through `protobuf`_ if possible.

    :exc:`~werkzeug.exceptions.BadRequest` will be raised if the content-type
    is not protobuf or if the data itself cannot be parsed property.

    .. _protobuf: http://code.google.com/p/protobuf/
    """

    #: by default the :class:`ProtobufRequestMixin` will raise a
    #: :exc:`~werkzeug.exceptions.BadRequest` if the object is not
    #: initialized.  You can bypass that check by setting this
    #: attribute to `False`.
    protobuf_check_initialization = True

    def parse_protobuf(self, proto_type):
        """Parse the data into an instance of proto_type."""
        if 'protobuf' not in self.environ.get('CONTENT_TYPE', ''):
            raise BadRequest('Not a Protobuf request')

        obj = proto_type()
        try:
            obj.ParseFromString(self.data)
        except Exception:
            raise BadRequest("Unable to parse Protobuf request")

        # Fail if not all required fields are set
        if self.protobuf_check_initialization and not obj.IsInitialized():
            raise BadRequest("Partial Protobuf request")

        return obj


class RoutingArgsRequestMixin(object):

    """This request mixin adds support for the wsgiorg routing args
    `specification`_.

    .. _specification: http://www.wsgi.org/wsgi/Specifications/routing_args
    """

    def _get_routing_args(self):
        return self.environ.get('wsgiorg.routing_args', (()))[0]

    def _set_routing_args(self, value):
        if self.shallow:
            raise RuntimeError('A shallow request tried to modify the WSGI '
                               'environment.  If you really want to do that, '
                               'set `shallow` to False.')
        self.environ['wsgiorg.routing_args'] = (value, self.routing_vars)

    routing_args = property(_get_routing_args, _set_routing_args, doc='''
        The positional URL arguments as `tuple`.''')
    del _get_routing_args, _set_routing_args

    def _get_routing_vars(self):
        rv = self.environ.get('wsgiorg.routing_args')
        if rv is not None:
            return rv[1]
        rv = {}
        if not self.shallow:
            self.routing_vars = rv
        return rv

    def _set_routing_vars(self, value):
        if self.shallow:
            raise RuntimeError('A shallow request tried to modify the WSGI '
                               'environment.  If you really want to do that, '
                               'set `shallow` to False.')
        self.environ['wsgiorg.routing_args'] = (self.routing_args, value)

    routing_vars = property(_get_routing_vars, _set_routing_vars, doc='''
        The keyword URL arguments as `dict`.''')
    del _get_routing_vars, _set_routing_vars


class ReverseSlashBehaviorRequestMixin(object):

    """This mixin reverses the trailing slash behavior of :attr:`script_root`
    and :attr:`path`.  This makes it possible to use :func:`~urlparse.urljoin`
    directly on the paths.

    Because it changes the behavior or :class:`Request` this class has to be
    mixed in *before* the actual request class::

        class MyRequest(ReverseSlashBehaviorRequestMixin, Request):
            pass

    This example shows the differences (for an application mounted on
    `/application` and the request going to `/application/foo/bar`):

        +---------------+-------------------+---------------------+
        |               | normal behavior   | reverse behavior    |
        +===============+===================+=====================+
        | `script_root` | ``/application``  | ``/application/``   |
        +---------------+-------------------+---------------------+
        | `path`        | ``/foo/bar``      | ``foo/bar``         |
        +---------------+-------------------+---------------------+
    """

    @cached_property
    def path(self):
        """Requested path as unicode.  This works a bit like the regular path
        info in the WSGI environment but will not include a leading slash.
        """
        path = wsgi_decoding_dance(self.environ.get('PATH_INFO') or '',
                                   self.charset, self.encoding_errors)
        return path.lstrip('/')

    @cached_property
    def script_root(self):
        """The root path of the script includling a trailing slash."""
        path = wsgi_decoding_dance(self.environ.get('SCRIPT_NAME') or '',
                                   self.charset, self.encoding_errors)
        return path.rstrip('/') + '/'


class DynamicCharsetRequestMixin(object):

    """"If this mixin is mixed into a request class it will provide
    a dynamic `charset` attribute.  This means that if the charset is
    transmitted in the content type headers it's used from there.

    Because it changes the behavior or :class:`Request` this class has
    to be mixed in *before* the actual request class::

        class MyRequest(DynamicCharsetRequestMixin, Request):
            pass

    By default the request object assumes that the URL charset is the
    same as the data charset.  If the charset varies on each request
    based on the transmitted data it's not a good idea to let the URLs
    change based on that.  Most browsers assume either utf-8 or latin1
    for the URLs if they have troubles figuring out.  It's strongly
    recommended to set the URL charset to utf-8::

        class MyRequest(DynamicCharsetRequestMixin, Request):
            url_charset = 'utf-8'

    .. versionadded:: 0.6
    """

    #: the default charset that is assumed if the content type header
    #: is missing or does not contain a charset parameter.  The default
    #: is latin1 which is what HTTP specifies as default charset.
    #: You may however want to set this to utf-8 to better support
    #: browsers that do not transmit a charset for incoming data.
    default_charset = 'latin1'

    def unknown_charset(self, charset):
        """Called if a charset was provided but is not supported by
        the Python codecs module.  By default latin1 is assumed then
        to not lose any information, you may override this method to
        change the behavior.

        :param charset: the charset that was not found.
        :return: the replacement charset.
        """
        return 'latin1'

    @cached_property
    def charset(self):
        """The charset from the content type."""
        header = self.environ.get('CONTENT_TYPE')
        if header:
            ct, options = parse_options_header(header)
            charset = options.get('charset')
            if charset:
                if is_known_charset(charset):
                    return charset
                return self.unknown_charset(charset)
        return self.default_charset


class DynamicCharsetResponseMixin(object):

    """If this mixin is mixed into a response class it will provide
    a dynamic `charset` attribute.  This means that if the charset is
    looked up and stored in the `Content-Type` header and updates
    itself automatically.  This also means a small performance hit but
    can be useful if you're working with different charsets on
    responses.

    Because the charset attribute is no a property at class-level, the
    default value is stored in `default_charset`.

    Because it changes the behavior or :class:`Response` this class has
    to be mixed in *before* the actual response class::

        class MyResponse(DynamicCharsetResponseMixin, Response):
            pass

    .. versionadded:: 0.6
    """

    #: the default charset.
    default_charset = 'utf-8'

    def _get_charset(self):
        header = self.headers.get('content-type')
        if header:
            charset = parse_options_header(header)[1].get('charset')
            if charset:
                return charset
        return self.default_charset

    def _set_charset(self, charset):
        header = self.headers.get('content-type')
        ct, options = parse_options_header(header)
        if not ct:
            raise TypeError('Cannot set charset if Content-Type '
                            'header is missing.')
        options['charset'] = charset
        self.headers['Content-Type'] = dump_options_header(ct, options)

    charset = property(_get_charset, _set_charset, doc="""
        The charset for the response.  It's stored inside the
        Content-Type header as a parameter.""")
    del _get_charset, _set_charset
