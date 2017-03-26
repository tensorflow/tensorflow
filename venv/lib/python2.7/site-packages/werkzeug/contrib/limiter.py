# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.limiter
    ~~~~~~~~~~~~~~~~~~~~~~~~

    A middleware that limits incoming data.  This works around problems with
    Trac_ or Django_ because those directly stream into the memory.

    .. _Trac: http://trac.edgewall.org/
    .. _Django: http://www.djangoproject.com/

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from warnings import warn

from werkzeug.wsgi import LimitedStream


class StreamLimitMiddleware(object):

    """Limits the input stream to a given number of bytes.  This is useful if
    you have a WSGI application that reads form data into memory (django for
    example) and you don't want users to harm the server by uploading tons of
    data.

    Default is 10MB

    .. versionchanged:: 0.9
       Deprecated middleware.
    """

    def __init__(self, app, maximum_size=1024 * 1024 * 10):
        warn(DeprecationWarning('This middleware is deprecated'))
        self.app = app
        self.maximum_size = maximum_size

    def __call__(self, environ, start_response):
        limit = min(self.maximum_size, int(environ.get('CONTENT_LENGTH') or 0))
        environ['wsgi.input'] = LimitedStream(environ['wsgi.input'], limit)
        return self.app(environ, start_response)
