# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.profiler
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This module provides a simple WSGI profiler middleware for finding
    bottlenecks in web application.  It uses the :mod:`profile` or
    :mod:`cProfile` module to do the profiling and writes the stats to the
    stream provided (defaults to stderr).

    Example usage::

        from werkzeug.contrib.profiler import ProfilerMiddleware
        app = ProfilerMiddleware(app)

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import sys
import time
import os.path
try:
    try:
        from cProfile import Profile
    except ImportError:
        from profile import Profile
    from pstats import Stats
    available = True
except ImportError:
    available = False


class MergeStream(object):

    """An object that redirects `write` calls to multiple streams.
    Use this to log to both `sys.stdout` and a file::

        f = open('profiler.log', 'w')
        stream = MergeStream(sys.stdout, f)
        profiler = ProfilerMiddleware(app, stream)
    """

    def __init__(self, *streams):
        if not streams:
            raise TypeError('at least one stream must be given')
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)


class ProfilerMiddleware(object):

    """Simple profiler middleware.  Wraps a WSGI application and profiles
    a request.  This intentionally buffers the response so that timings are
    more exact.

    By giving the `profile_dir` argument, pstat.Stats files are saved to that
    directory, one file per request. Without it, a summary is printed to
    `stream` instead.

    For the exact meaning of `sort_by` and `restrictions` consult the
    :mod:`profile` documentation.

    .. versionadded:: 0.9
       Added support for `restrictions` and `profile_dir`.

    :param app: the WSGI application to profile.
    :param stream: the stream for the profiled stats.  defaults to stderr.
    :param sort_by: a tuple of columns to sort the result by.
    :param restrictions: a tuple of profiling strictions, not used if dumping
                         to `profile_dir`.
    :param profile_dir: directory name to save pstat files
    """

    def __init__(self, app, stream=None,
                 sort_by=('time', 'calls'), restrictions=(), profile_dir=None):
        if not available:
            raise RuntimeError('the profiler is not available because '
                               'profile or pstat is not installed.')
        self._app = app
        self._stream = stream or sys.stdout
        self._sort_by = sort_by
        self._restrictions = restrictions
        self._profile_dir = profile_dir

    def __call__(self, environ, start_response):
        response_body = []

        def catching_start_response(status, headers, exc_info=None):
            start_response(status, headers, exc_info)
            return response_body.append

        def runapp():
            appiter = self._app(environ, catching_start_response)
            response_body.extend(appiter)
            if hasattr(appiter, 'close'):
                appiter.close()

        p = Profile()
        start = time.time()
        p.runcall(runapp)
        body = b''.join(response_body)
        elapsed = time.time() - start

        if self._profile_dir is not None:
            prof_filename = os.path.join(self._profile_dir,
                                         '%s.%s.%06dms.%d.prof' % (
                                             environ['REQUEST_METHOD'],
                                             environ.get('PATH_INFO').strip(
                                                 '/').replace('/', '.') or 'root',
                                             elapsed * 1000.0,
                                             time.time()
                                         ))
            p.dump_stats(prof_filename)

        else:
            stats = Stats(p, stream=self._stream)
            stats.sort_stats(*self._sort_by)

            self._stream.write('-' * 80)
            self._stream.write('\nPATH: %r\n' % environ.get('PATH_INFO'))
            stats.print_stats(*self._restrictions)
            self._stream.write('-' * 80 + '\n\n')

        return [body]


def make_action(app_factory, hostname='localhost', port=5000,
                threaded=False, processes=1, stream=None,
                sort_by=('time', 'calls'), restrictions=()):
    """Return a new callback for :mod:`werkzeug.script` that starts a local
    server with the profiler enabled.

    ::

        from werkzeug.contrib import profiler
        action_profile = profiler.make_action(make_app)
    """
    def action(hostname=('h', hostname), port=('p', port),
               threaded=threaded, processes=processes):
        """Start a new development server."""
        from werkzeug.serving import run_simple
        app = ProfilerMiddleware(app_factory(), stream, sort_by, restrictions)
        run_simple(hostname, port, app, False, None, threaded, processes)
    return action
