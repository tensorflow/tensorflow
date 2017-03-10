# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.fixers
    ~~~~~~~~~~~~~~~~~~~~~~~

    .. versionadded:: 0.5

    This module includes various helpers that fix bugs in web servers.  They may
    be necessary for some versions of a buggy web server but not others.  We try
    to stay updated with the status of the bugs as good as possible but you have
    to make sure whether they fix the problem you encounter.

    If you notice bugs in webservers not fixed in this module consider
    contributing a patch.

    :copyright: Copyright 2009 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote

from werkzeug.http import parse_options_header, parse_cache_control_header, \
    parse_set_header
from werkzeug.useragents import UserAgent
from werkzeug.datastructures import Headers, ResponseCacheControl


class CGIRootFix(object):

    """Wrap the application in this middleware if you are using FastCGI or CGI
    and you have problems with your app root being set to the cgi script's path
    instead of the path users are going to visit

    .. versionchanged:: 0.9
       Added `app_root` parameter and renamed from `LighttpdCGIRootFix`.

    :param app: the WSGI application
    :param app_root: Defaulting to ``'/'``, you can set this to something else
        if your app is mounted somewhere else.
    """

    def __init__(self, app, app_root='/'):
        self.app = app
        self.app_root = app_root

    def __call__(self, environ, start_response):
        # only set PATH_INFO for older versions of Lighty or if no
        # server software is provided.  That's because the test was
        # added in newer Werkzeug versions and we don't want to break
        # people's code if they are using this fixer in a test that
        # does not set the SERVER_SOFTWARE key.
        if 'SERVER_SOFTWARE' not in environ or \
           environ['SERVER_SOFTWARE'] < 'lighttpd/1.4.28':
            environ['PATH_INFO'] = environ.get('SCRIPT_NAME', '') + \
                environ.get('PATH_INFO', '')
        environ['SCRIPT_NAME'] = self.app_root.strip('/')
        return self.app(environ, start_response)

# backwards compatibility
LighttpdCGIRootFix = CGIRootFix


class PathInfoFromRequestUriFix(object):

    """On windows environment variables are limited to the system charset
    which makes it impossible to store the `PATH_INFO` variable in the
    environment without loss of information on some systems.

    This is for example a problem for CGI scripts on a Windows Apache.

    This fixer works by recreating the `PATH_INFO` from `REQUEST_URI`,
    `REQUEST_URL`, or `UNENCODED_URL` (whatever is available).  Thus the
    fix can only be applied if the webserver supports either of these
    variables.

    :param app: the WSGI application
    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        for key in 'REQUEST_URL', 'REQUEST_URI', 'UNENCODED_URL':
            if key not in environ:
                continue
            request_uri = unquote(environ[key])
            script_name = unquote(environ.get('SCRIPT_NAME', ''))
            if request_uri.startswith(script_name):
                environ['PATH_INFO'] = request_uri[len(script_name):] \
                    .split('?', 1)[0]
                break
        return self.app(environ, start_response)


class ProxyFix(object):

    """This middleware can be applied to add HTTP proxy support to an
    application that was not designed with HTTP proxies in mind.  It
    sets `REMOTE_ADDR`, `HTTP_HOST` from `X-Forwarded` headers.  While
    Werkzeug-based applications already can use
    :py:func:`werkzeug.wsgi.get_host` to retrieve the current host even if
    behind proxy setups, this middleware can be used for applications which
    access the WSGI environment directly.

    If you have more than one proxy server in front of your app, set
    `num_proxies` accordingly.

    Do not use this middleware in non-proxy setups for security reasons.

    The original values of `REMOTE_ADDR` and `HTTP_HOST` are stored in
    the WSGI environment as `werkzeug.proxy_fix.orig_remote_addr` and
    `werkzeug.proxy_fix.orig_http_host`.

    :param app: the WSGI application
    :param num_proxies: the number of proxy servers in front of the app.
    """

    def __init__(self, app, num_proxies=1):
        self.app = app
        self.num_proxies = num_proxies

    def get_remote_addr(self, forwarded_for):
        """Selects the new remote addr from the given list of ips in
        X-Forwarded-For.  By default it picks the one that the `num_proxies`
        proxy server provides.  Before 0.9 it would always pick the first.

        .. versionadded:: 0.8
        """
        if len(forwarded_for) >= self.num_proxies:
            return forwarded_for[-1 * self.num_proxies]

    def __call__(self, environ, start_response):
        getter = environ.get
        forwarded_proto = getter('HTTP_X_FORWARDED_PROTO', '')
        forwarded_for = getter('HTTP_X_FORWARDED_FOR', '').split(',')
        forwarded_host = getter('HTTP_X_FORWARDED_HOST', '')
        environ.update({
            'werkzeug.proxy_fix.orig_wsgi_url_scheme':  getter('wsgi.url_scheme'),
            'werkzeug.proxy_fix.orig_remote_addr':      getter('REMOTE_ADDR'),
            'werkzeug.proxy_fix.orig_http_host':        getter('HTTP_HOST')
        })
        forwarded_for = [x for x in [x.strip() for x in forwarded_for] if x]
        remote_addr = self.get_remote_addr(forwarded_for)
        if remote_addr is not None:
            environ['REMOTE_ADDR'] = remote_addr
        if forwarded_host:
            environ['HTTP_HOST'] = forwarded_host
        if forwarded_proto:
            environ['wsgi.url_scheme'] = forwarded_proto
        return self.app(environ, start_response)


class HeaderRewriterFix(object):

    """This middleware can remove response headers and add others.  This
    is for example useful to remove the `Date` header from responses if you
    are using a server that adds that header, no matter if it's present or
    not or to add `X-Powered-By` headers::

        app = HeaderRewriterFix(app, remove_headers=['Date'],
                                add_headers=[('X-Powered-By', 'WSGI')])

    :param app: the WSGI application
    :param remove_headers: a sequence of header keys that should be
                           removed.
    :param add_headers: a sequence of ``(key, value)`` tuples that should
                        be added.
    """

    def __init__(self, app, remove_headers=None, add_headers=None):
        self.app = app
        self.remove_headers = set(x.lower() for x in (remove_headers or ()))
        self.add_headers = list(add_headers or ())

    def __call__(self, environ, start_response):
        def rewriting_start_response(status, headers, exc_info=None):
            new_headers = []
            for key, value in headers:
                if key.lower() not in self.remove_headers:
                    new_headers.append((key, value))
            new_headers += self.add_headers
            return start_response(status, new_headers, exc_info)
        return self.app(environ, rewriting_start_response)


class InternetExplorerFix(object):

    """This middleware fixes a couple of bugs with Microsoft Internet
    Explorer.  Currently the following fixes are applied:

    -   removing of `Vary` headers for unsupported mimetypes which
        causes troubles with caching.  Can be disabled by passing
        ``fix_vary=False`` to the constructor.
        see: http://support.microsoft.com/kb/824847/en-us

    -   removes offending headers to work around caching bugs in
        Internet Explorer if `Content-Disposition` is set.  Can be
        disabled by passing ``fix_attach=False`` to the constructor.

    If it does not detect affected Internet Explorer versions it won't touch
    the request / response.
    """

    # This code was inspired by Django fixers for the same bugs.  The
    # fix_vary and fix_attach fixers were originally implemented in Django
    # by Michael Axiak and is available as part of the Django project:
    #     http://code.djangoproject.com/ticket/4148

    def __init__(self, app, fix_vary=True, fix_attach=True):
        self.app = app
        self.fix_vary = fix_vary
        self.fix_attach = fix_attach

    def fix_headers(self, environ, headers, status=None):
        if self.fix_vary:
            header = headers.get('content-type', '')
            mimetype, options = parse_options_header(header)
            if mimetype not in ('text/html', 'text/plain', 'text/sgml'):
                headers.pop('vary', None)

        if self.fix_attach and 'content-disposition' in headers:
            pragma = parse_set_header(headers.get('pragma', ''))
            pragma.discard('no-cache')
            header = pragma.to_header()
            if not header:
                headers.pop('pragma', '')
            else:
                headers['Pragma'] = header
            header = headers.get('cache-control', '')
            if header:
                cc = parse_cache_control_header(header,
                                                cls=ResponseCacheControl)
                cc.no_cache = None
                cc.no_store = False
                header = cc.to_header()
                if not header:
                    headers.pop('cache-control', '')
                else:
                    headers['Cache-Control'] = header

    def run_fixed(self, environ, start_response):
        def fixing_start_response(status, headers, exc_info=None):
            headers = Headers(headers)
            self.fix_headers(environ, headers, status)
            return start_response(status, headers.to_wsgi_list(), exc_info)
        return self.app(environ, fixing_start_response)

    def __call__(self, environ, start_response):
        ua = UserAgent(environ)
        if ua.browser != 'msie':
            return self.app(environ, start_response)
        return self.run_fixed(environ, start_response)
