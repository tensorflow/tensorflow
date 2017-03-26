# -*- coding: utf-8 -*-
"""
    flask.testing
    ~~~~~~~~~~~~~

    Implements test support helpers.  This module is lazily imported
    and usually not used in production environments.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

import werkzeug
from contextlib import contextmanager
from werkzeug.test import Client, EnvironBuilder
from flask import _request_ctx_stack

try:
    from werkzeug.urls import url_parse
except ImportError:
    from urlparse import urlsplit as url_parse


def make_test_environ_builder(app, path='/', base_url=None, *args, **kwargs):
    """Creates a new test builder with some application defaults thrown in."""
    http_host = app.config.get('SERVER_NAME')
    app_root = app.config.get('APPLICATION_ROOT')
    if base_url is None:
        url = url_parse(path)
        base_url = 'http://%s/' % (url.netloc or http_host or 'localhost')
        if app_root:
            base_url += app_root.lstrip('/')
        if url.netloc:
            path = url.path
            if url.query:
                path += '?' + url.query
    return EnvironBuilder(path, base_url, *args, **kwargs)


class FlaskClient(Client):
    """Works like a regular Werkzeug test client but has some knowledge about
    how Flask works to defer the cleanup of the request context stack to the
    end of a ``with`` body when used in a ``with`` statement.  For general
    information about how to use this class refer to
    :class:`werkzeug.test.Client`.

    .. versionchanged:: 0.12
       `app.test_client()` includes preset default environment, which can be
       set after instantiation of the `app.test_client()` object in
       `client.environ_base`.

    Basic usage is outlined in the :ref:`testing` chapter.
    """

    preserve_context = False

    def __init__(self, *args, **kwargs):
        super(FlaskClient, self).__init__(*args, **kwargs)
        self.environ_base = {
            "REMOTE_ADDR": "127.0.0.1",
            "HTTP_USER_AGENT": "werkzeug/" + werkzeug.__version__
        }

    @contextmanager
    def session_transaction(self, *args, **kwargs):
        """When used in combination with a ``with`` statement this opens a
        session transaction.  This can be used to modify the session that
        the test client uses.  Once the ``with`` block is left the session is
        stored back.

        ::

            with client.session_transaction() as session:
                session['value'] = 42

        Internally this is implemented by going through a temporary test
        request context and since session handling could depend on
        request variables this function accepts the same arguments as
        :meth:`~flask.Flask.test_request_context` which are directly
        passed through.
        """
        if self.cookie_jar is None:
            raise RuntimeError('Session transactions only make sense '
                               'with cookies enabled.')
        app = self.application
        environ_overrides = kwargs.setdefault('environ_overrides', {})
        self.cookie_jar.inject_wsgi(environ_overrides)
        outer_reqctx = _request_ctx_stack.top
        with app.test_request_context(*args, **kwargs) as c:
            sess = app.open_session(c.request)
            if sess is None:
                raise RuntimeError('Session backend did not open a session. '
                                   'Check the configuration')

            # Since we have to open a new request context for the session
            # handling we want to make sure that we hide out own context
            # from the caller.  By pushing the original request context
            # (or None) on top of this and popping it we get exactly that
            # behavior.  It's important to not use the push and pop
            # methods of the actual request context object since that would
            # mean that cleanup handlers are called
            _request_ctx_stack.push(outer_reqctx)
            try:
                yield sess
            finally:
                _request_ctx_stack.pop()

            resp = app.response_class()
            if not app.session_interface.is_null_session(sess):
                app.save_session(sess, resp)
            headers = resp.get_wsgi_headers(c.request.environ)
            self.cookie_jar.extract_wsgi(c.request.environ, headers)

    def open(self, *args, **kwargs):
        kwargs.setdefault('environ_overrides', {}) \
            ['flask._preserve_context'] = self.preserve_context
        kwargs.setdefault('environ_base', self.environ_base)

        as_tuple = kwargs.pop('as_tuple', False)
        buffered = kwargs.pop('buffered', False)
        follow_redirects = kwargs.pop('follow_redirects', False)
        builder = make_test_environ_builder(self.application, *args, **kwargs)

        return Client.open(self, builder,
                           as_tuple=as_tuple,
                           buffered=buffered,
                           follow_redirects=follow_redirects)

    def __enter__(self):
        if self.preserve_context:
            raise RuntimeError('Cannot nest client invocations')
        self.preserve_context = True
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.preserve_context = False

        # on exit we want to clean up earlier.  Normally the request context
        # stays preserved until the next request in the same thread comes
        # in.  See RequestGlobals.push() for the general behavior.
        top = _request_ctx_stack.top
        if top is not None and top.preserved:
            top.pop()
