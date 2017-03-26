# -*- coding: utf-8 -*-
"""
    flask.helpers
    ~~~~~~~~~~~~~

    Implements various helpers.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

import os
import sys
import pkgutil
import posixpath
import mimetypes
from time import time
from zlib import adler32
from threading import RLock
from werkzeug.routing import BuildError
from functools import update_wrapper

try:
    from werkzeug.urls import url_quote
except ImportError:
    from urlparse import quote as url_quote

from werkzeug.datastructures import Headers, Range
from werkzeug.exceptions import BadRequest, NotFound, \
    RequestedRangeNotSatisfiable

# this was moved in 0.7
try:
    from werkzeug.wsgi import wrap_file
except ImportError:
    from werkzeug.utils import wrap_file

from jinja2 import FileSystemLoader

from .signals import message_flashed
from .globals import session, _request_ctx_stack, _app_ctx_stack, \
     current_app, request
from ._compat import string_types, text_type


# sentinel
_missing = object()


# what separators does this operating system provide that are not a slash?
# this is used by the send_from_directory function to ensure that nobody is
# able to access files from outside the filesystem.
_os_alt_seps = list(sep for sep in [os.path.sep, os.path.altsep]
                    if sep not in (None, '/'))


def get_debug_flag(default=None):
    val = os.environ.get('FLASK_DEBUG')
    if not val:
        return default
    return val not in ('0', 'false', 'no')


def _endpoint_from_view_func(view_func):
    """Internal helper that returns the default endpoint for a given
    function.  This always is the function name.
    """
    assert view_func is not None, 'expected view func if endpoint ' \
                                  'is not provided.'
    return view_func.__name__


def stream_with_context(generator_or_function):
    """Request contexts disappear when the response is started on the server.
    This is done for efficiency reasons and to make it less likely to encounter
    memory leaks with badly written WSGI middlewares.  The downside is that if
    you are using streamed responses, the generator cannot access request bound
    information any more.

    This function however can help you keep the context around for longer::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            @stream_with_context
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(generate())

    Alternatively it can also be used around a specific generator::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(stream_with_context(generate()))

    .. versionadded:: 0.9
    """
    try:
        gen = iter(generator_or_function)
    except TypeError:
        def decorator(*args, **kwargs):
            gen = generator_or_function(*args, **kwargs)
            return stream_with_context(gen)
        return update_wrapper(decorator, generator_or_function)

    def generator():
        ctx = _request_ctx_stack.top
        if ctx is None:
            raise RuntimeError('Attempted to stream with context but '
                'there was no context in the first place to keep around.')
        with ctx:
            # Dummy sentinel.  Has to be inside the context block or we're
            # not actually keeping the context around.
            yield None

            # The try/finally is here so that if someone passes a WSGI level
            # iterator in we're still running the cleanup logic.  Generators
            # don't need that because they are closed on their destruction
            # automatically.
            try:
                for item in gen:
                    yield item
            finally:
                if hasattr(gen, 'close'):
                    gen.close()

    # The trick is to start the generator.  Then the code execution runs until
    # the first dummy None is yielded at which point the context was already
    # pushed.  This item is discarded.  Then when the iteration continues the
    # real generator is executed.
    wrapped_g = generator()
    next(wrapped_g)
    return wrapped_g


def make_response(*args):
    """Sometimes it is necessary to set additional headers in a view.  Because
    views do not have to return response objects but can return a value that
    is converted into a response object by Flask itself, it becomes tricky to
    add headers to it.  This function can be called instead of using a return
    and you will get a response object which you can use to attach headers.

    If view looked like this and you want to add a new header::

        def index():
            return render_template('index.html', foo=42)

    You can now do something like this::

        def index():
            response = make_response(render_template('index.html', foo=42))
            response.headers['X-Parachutes'] = 'parachutes are cool'
            return response

    This function accepts the very same arguments you can return from a
    view function.  This for example creates a response with a 404 error
    code::

        response = make_response(render_template('not_found.html'), 404)

    The other use case of this function is to force the return value of a
    view function into a response which is helpful with view
    decorators::

        response = make_response(view_function())
        response.headers['X-Parachutes'] = 'parachutes are cool'

    Internally this function does the following things:

    -   if no arguments are passed, it creates a new response argument
    -   if one argument is passed, :meth:`flask.Flask.make_response`
        is invoked with it.
    -   if more than one argument is passed, the arguments are passed
        to the :meth:`flask.Flask.make_response` function as tuple.

    .. versionadded:: 0.6
    """
    if not args:
        return current_app.response_class()
    if len(args) == 1:
        args = args[0]
    return current_app.make_response(args)


def url_for(endpoint, **values):
    """Generates a URL to the given endpoint with the method provided.

    Variable arguments that are unknown to the target endpoint are appended
    to the generated URL as query arguments.  If the value of a query argument
    is ``None``, the whole pair is skipped.  In case blueprints are active
    you can shortcut references to the same blueprint by prefixing the
    local endpoint with a dot (``.``).

    This will reference the index function local to the current blueprint::

        url_for('.index')

    For more information, head over to the :ref:`Quickstart <url-building>`.

    To integrate applications, :class:`Flask` has a hook to intercept URL build
    errors through :attr:`Flask.url_build_error_handlers`.  The `url_for`
    function results in a :exc:`~werkzeug.routing.BuildError` when the current
    app does not have a URL for the given endpoint and values.  When it does, the
    :data:`~flask.current_app` calls its :attr:`~Flask.url_build_error_handlers` if
    it is not ``None``, which can return a string to use as the result of
    `url_for` (instead of `url_for`'s default to raise the
    :exc:`~werkzeug.routing.BuildError` exception) or re-raise the exception.
    An example::

        def external_url_handler(error, endpoint, values):
            "Looks up an external URL when `url_for` cannot build a URL."
            # This is an example of hooking the build_error_handler.
            # Here, lookup_url is some utility function you've built
            # which looks up the endpoint in some external URL registry.
            url = lookup_url(endpoint, **values)
            if url is None:
                # External lookup did not have a URL.
                # Re-raise the BuildError, in context of original traceback.
                exc_type, exc_value, tb = sys.exc_info()
                if exc_value is error:
                    raise exc_type, exc_value, tb
                else:
                    raise error
            # url_for will use this result, instead of raising BuildError.
            return url

        app.url_build_error_handlers.append(external_url_handler)

    Here, `error` is the instance of :exc:`~werkzeug.routing.BuildError`, and
    `endpoint` and `values` are the arguments passed into `url_for`.  Note
    that this is for building URLs outside the current application, and not for
    handling 404 NotFound errors.

    .. versionadded:: 0.10
       The `_scheme` parameter was added.

    .. versionadded:: 0.9
       The `_anchor` and `_method` parameters were added.

    .. versionadded:: 0.9
       Calls :meth:`Flask.handle_build_error` on
       :exc:`~werkzeug.routing.BuildError`.

    :param endpoint: the endpoint of the URL (name of the function)
    :param values: the variable arguments of the URL rule
    :param _external: if set to ``True``, an absolute URL is generated. Server
      address can be changed via ``SERVER_NAME`` configuration variable which
      defaults to `localhost`.
    :param _scheme: a string specifying the desired URL scheme. The `_external`
      parameter must be set to ``True`` or a :exc:`ValueError` is raised. The default
      behavior uses the same scheme as the current request, or
      ``PREFERRED_URL_SCHEME`` from the :ref:`app configuration <config>` if no
      request context is available. As of Werkzeug 0.10, this also can be set
      to an empty string to build protocol-relative URLs.
    :param _anchor: if provided this is added as anchor to the URL.
    :param _method: if provided this explicitly specifies an HTTP method.
    """
    appctx = _app_ctx_stack.top
    reqctx = _request_ctx_stack.top
    if appctx is None:
        raise RuntimeError('Attempted to generate a URL without the '
                           'application context being pushed. This has to be '
                           'executed when application context is available.')

    # If request specific information is available we have some extra
    # features that support "relative" URLs.
    if reqctx is not None:
        url_adapter = reqctx.url_adapter
        blueprint_name = request.blueprint
        if not reqctx.request._is_old_module:
            if endpoint[:1] == '.':
                if blueprint_name is not None:
                    endpoint = blueprint_name + endpoint
                else:
                    endpoint = endpoint[1:]
        else:
            # TODO: get rid of this deprecated functionality in 1.0
            if '.' not in endpoint:
                if blueprint_name is not None:
                    endpoint = blueprint_name + '.' + endpoint
            elif endpoint.startswith('.'):
                endpoint = endpoint[1:]
        external = values.pop('_external', False)

    # Otherwise go with the url adapter from the appctx and make
    # the URLs external by default.
    else:
        url_adapter = appctx.url_adapter
        if url_adapter is None:
            raise RuntimeError('Application was not able to create a URL '
                               'adapter for request independent URL generation. '
                               'You might be able to fix this by setting '
                               'the SERVER_NAME config variable.')
        external = values.pop('_external', True)

    anchor = values.pop('_anchor', None)
    method = values.pop('_method', None)
    scheme = values.pop('_scheme', None)
    appctx.app.inject_url_defaults(endpoint, values)

    # This is not the best way to deal with this but currently the
    # underlying Werkzeug router does not support overriding the scheme on
    # a per build call basis.
    old_scheme = None
    if scheme is not None:
        if not external:
            raise ValueError('When specifying _scheme, _external must be True')
        old_scheme = url_adapter.url_scheme
        url_adapter.url_scheme = scheme

    try:
        try:
            rv = url_adapter.build(endpoint, values, method=method,
                                   force_external=external)
        finally:
            if old_scheme is not None:
                url_adapter.url_scheme = old_scheme
    except BuildError as error:
        # We need to inject the values again so that the app callback can
        # deal with that sort of stuff.
        values['_external'] = external
        values['_anchor'] = anchor
        values['_method'] = method
        return appctx.app.handle_url_build_error(error, endpoint, values)

    if anchor is not None:
        rv += '#' + url_quote(anchor)
    return rv


def get_template_attribute(template_name, attribute):
    """Loads a macro (or variable) a template exports.  This can be used to
    invoke a macro from within Python code.  If you for example have a
    template named :file:`_cider.html` with the following contents:

    .. sourcecode:: html+jinja

       {% macro hello(name) %}Hello {{ name }}!{% endmacro %}

    You can access this from Python code like this::

        hello = get_template_attribute('_cider.html', 'hello')
        return hello('World')

    .. versionadded:: 0.2

    :param template_name: the name of the template
    :param attribute: the name of the variable of macro to access
    """
    return getattr(current_app.jinja_env.get_template(template_name).module,
                   attribute)


def flash(message, category='message'):
    """Flashes a message to the next request.  In order to remove the
    flashed message from the session and to display it to the user,
    the template has to call :func:`get_flashed_messages`.

    .. versionchanged:: 0.3
       `category` parameter added.

    :param message: the message to be flashed.
    :param category: the category for the message.  The following values
                     are recommended: ``'message'`` for any kind of message,
                     ``'error'`` for errors, ``'info'`` for information
                     messages and ``'warning'`` for warnings.  However any
                     kind of string can be used as category.
    """
    # Original implementation:
    #
    #     session.setdefault('_flashes', []).append((category, message))
    #
    # This assumed that changes made to mutable structures in the session are
    # are always in sync with the session object, which is not true for session
    # implementations that use external storage for keeping their keys/values.
    flashes = session.get('_flashes', [])
    flashes.append((category, message))
    session['_flashes'] = flashes
    message_flashed.send(current_app._get_current_object(),
                         message=message, category=category)


def get_flashed_messages(with_categories=False, category_filter=[]):
    """Pulls all flashed messages from the session and returns them.
    Further calls in the same request to the function will return
    the same messages.  By default just the messages are returned,
    but when `with_categories` is set to ``True``, the return value will
    be a list of tuples in the form ``(category, message)`` instead.

    Filter the flashed messages to one or more categories by providing those
    categories in `category_filter`.  This allows rendering categories in
    separate html blocks.  The `with_categories` and `category_filter`
    arguments are distinct:

    * `with_categories` controls whether categories are returned with message
      text (``True`` gives a tuple, where ``False`` gives just the message text).
    * `category_filter` filters the messages down to only those matching the
      provided categories.

    See :ref:`message-flashing-pattern` for examples.

    .. versionchanged:: 0.3
       `with_categories` parameter added.

    .. versionchanged:: 0.9
        `category_filter` parameter added.

    :param with_categories: set to ``True`` to also receive categories.
    :param category_filter: whitelist of categories to limit return values
    """
    flashes = _request_ctx_stack.top.flashes
    if flashes is None:
        _request_ctx_stack.top.flashes = flashes = session.pop('_flashes') \
            if '_flashes' in session else []
    if category_filter:
        flashes = list(filter(lambda f: f[0] in category_filter, flashes))
    if not with_categories:
        return [x[1] for x in flashes]
    return flashes


def send_file(filename_or_fp, mimetype=None, as_attachment=False,
              attachment_filename=None, add_etags=True,
              cache_timeout=None, conditional=False, last_modified=None):
    """Sends the contents of a file to the client.  This will use the
    most efficient method available and configured.  By default it will
    try to use the WSGI server's file_wrapper support.  Alternatively
    you can set the application's :attr:`~Flask.use_x_sendfile` attribute
    to ``True`` to directly emit an ``X-Sendfile`` header.  This however
    requires support of the underlying webserver for ``X-Sendfile``.

    By default it will try to guess the mimetype for you, but you can
    also explicitly provide one.  For extra security you probably want
    to send certain files as attachment (HTML for instance).  The mimetype
    guessing requires a `filename` or an `attachment_filename` to be
    provided.

    ETags will also be attached automatically if a `filename` is provided. You
    can turn this off by setting `add_etags=False`.

    If `conditional=True` and `filename` is provided, this method will try to
    upgrade the response stream to support range requests.  This will allow
    the request to be answered with partial content response.

    Please never pass filenames to this function from user sources;
    you should use :func:`send_from_directory` instead.

    .. versionadded:: 0.2

    .. versionadded:: 0.5
       The `add_etags`, `cache_timeout` and `conditional` parameters were
       added.  The default behavior is now to attach etags.

    .. versionchanged:: 0.7
       mimetype guessing and etag support for file objects was
       deprecated because it was unreliable.  Pass a filename if you are
       able to, otherwise attach an etag yourself.  This functionality
       will be removed in Flask 1.0

    .. versionchanged:: 0.9
       cache_timeout pulls its default from application config, when None.

    .. versionchanged:: 0.12
       The filename is no longer automatically inferred from file objects. If
       you want to use automatic mimetype and etag support, pass a filepath via
       `filename_or_fp` or `attachment_filename`.

    .. versionchanged:: 0.12
       The `attachment_filename` is preferred over `filename` for MIME-type
       detection.

    :param filename_or_fp: the filename of the file to send in `latin-1`.
                           This is relative to the :attr:`~Flask.root_path`
                           if a relative path is specified.
                           Alternatively a file object might be provided in
                           which case ``X-Sendfile`` might not work and fall
                           back to the traditional method.  Make sure that the
                           file pointer is positioned at the start of data to
                           send before calling :func:`send_file`.
    :param mimetype: the mimetype of the file if provided. If a file path is
                     given, auto detection happens as fallback, otherwise an
                     error will be raised.
    :param as_attachment: set to ``True`` if you want to send this file with
                          a ``Content-Disposition: attachment`` header.
    :param attachment_filename: the filename for the attachment if it
                                differs from the file's filename.
    :param add_etags: set to ``False`` to disable attaching of etags.
    :param conditional: set to ``True`` to enable conditional responses.

    :param cache_timeout: the timeout in seconds for the headers. When ``None``
                          (default), this value is set by
                          :meth:`~Flask.get_send_file_max_age` of
                          :data:`~flask.current_app`.
    :param last_modified: set the ``Last-Modified`` header to this value,
        a :class:`~datetime.datetime` or timestamp.
        If a file was passed, this overrides its mtime.
    """
    mtime = None
    fsize = None
    if isinstance(filename_or_fp, string_types):
        filename = filename_or_fp
        if not os.path.isabs(filename):
            filename = os.path.join(current_app.root_path, filename)
        file = None
        if attachment_filename is None:
            attachment_filename = os.path.basename(filename)
    else:
        file = filename_or_fp
        filename = None

    if mimetype is None:
        if attachment_filename is not None:
            mimetype = mimetypes.guess_type(attachment_filename)[0] \
                or 'application/octet-stream'

        if mimetype is None:
            raise ValueError(
                'Unable to infer MIME-type because no filename is available. '
                'Please set either `attachment_filename`, pass a filepath to '
                '`filename_or_fp` or set your own MIME-type via `mimetype`.'
            )

    headers = Headers()
    if as_attachment:
        if attachment_filename is None:
            raise TypeError('filename unavailable, required for '
                            'sending as attachment')
        headers.add('Content-Disposition', 'attachment',
                    filename=attachment_filename)

    if current_app.use_x_sendfile and filename:
        if file is not None:
            file.close()
        headers['X-Sendfile'] = filename
        fsize = os.path.getsize(filename)
        headers['Content-Length'] = fsize
        data = None
    else:
        if file is None:
            file = open(filename, 'rb')
            mtime = os.path.getmtime(filename)
            fsize = os.path.getsize(filename)
            headers['Content-Length'] = fsize
        data = wrap_file(request.environ, file)

    rv = current_app.response_class(data, mimetype=mimetype, headers=headers,
                                    direct_passthrough=True)

    if last_modified is not None:
        rv.last_modified = last_modified
    elif mtime is not None:
        rv.last_modified = mtime

    rv.cache_control.public = True
    if cache_timeout is None:
        cache_timeout = current_app.get_send_file_max_age(filename)
    if cache_timeout is not None:
        rv.cache_control.max_age = cache_timeout
        rv.expires = int(time() + cache_timeout)

    if add_etags and filename is not None:
        from warnings import warn

        try:
            rv.set_etag('%s-%s-%s' % (
                os.path.getmtime(filename),
                os.path.getsize(filename),
                adler32(
                    filename.encode('utf-8') if isinstance(filename, text_type)
                    else filename
                ) & 0xffffffff
            ))
        except OSError:
            warn('Access %s failed, maybe it does not exist, so ignore etags in '
                 'headers' % filename, stacklevel=2)

    if conditional:
        if callable(getattr(Range, 'to_content_range_header', None)):
            # Werkzeug supports Range Requests
            # Remove this test when support for Werkzeug <0.12 is dropped
            try:
                rv = rv.make_conditional(request, accept_ranges=True,
                                         complete_length=fsize)
            except RequestedRangeNotSatisfiable:
                file.close()
                raise
        else:
            rv = rv.make_conditional(request)
        # make sure we don't send x-sendfile for servers that
        # ignore the 304 status code for x-sendfile.
        if rv.status_code == 304:
            rv.headers.pop('x-sendfile', None)
    return rv


def safe_join(directory, *pathnames):
    """Safely join `directory` and zero or more untrusted `pathnames`
    components.

    Example usage::

        @app.route('/wiki/<path:filename>')
        def wiki_page(filename):
            filename = safe_join(app.config['WIKI_FOLDER'], filename)
            with open(filename, 'rb') as fd:
                content = fd.read()  # Read and process the file content...

    :param directory: the trusted base directory.
    :param pathnames: the untrusted pathnames relative to that directory.
    :raises: :class:`~werkzeug.exceptions.NotFound` if one or more passed
            paths fall out of its boundaries.
    """
    for filename in pathnames:
        if filename != '':
            filename = posixpath.normpath(filename)
        for sep in _os_alt_seps:
            if sep in filename:
                raise NotFound()
        if os.path.isabs(filename) or \
           filename == '..' or \
           filename.startswith('../'):
            raise NotFound()
        directory = os.path.join(directory, filename)
    return directory


def send_from_directory(directory, filename, **options):
    """Send a file from a given directory with :func:`send_file`.  This
    is a secure way to quickly expose static files from an upload folder
    or something similar.

    Example usage::

        @app.route('/uploads/<path:filename>')
        def download_file(filename):
            return send_from_directory(app.config['UPLOAD_FOLDER'],
                                       filename, as_attachment=True)

    .. admonition:: Sending files and Performance

       It is strongly recommended to activate either ``X-Sendfile`` support in
       your webserver or (if no authentication happens) to tell the webserver
       to serve files for the given path on its own without calling into the
       web application for improved performance.

    .. versionadded:: 0.5

    :param directory: the directory where all the files are stored.
    :param filename: the filename relative to that directory to
                     download.
    :param options: optional keyword arguments that are directly
                    forwarded to :func:`send_file`.
    """
    filename = safe_join(directory, filename)
    if not os.path.isabs(filename):
        filename = os.path.join(current_app.root_path, filename)
    try:
        if not os.path.isfile(filename):
            raise NotFound()
    except (TypeError, ValueError):
        raise BadRequest()
    options.setdefault('conditional', True)
    return send_file(filename, **options)


def get_root_path(import_name):
    """Returns the path to a package or cwd if that cannot be found.  This
    returns the path of a package or the folder that contains a module.

    Not to be confused with the package path returned by :func:`find_package`.
    """
    # Module already imported and has a file attribute.  Use that first.
    mod = sys.modules.get(import_name)
    if mod is not None and hasattr(mod, '__file__'):
        return os.path.dirname(os.path.abspath(mod.__file__))

    # Next attempt: check the loader.
    loader = pkgutil.get_loader(import_name)

    # Loader does not exist or we're referring to an unloaded main module
    # or a main module without path (interactive sessions), go with the
    # current working directory.
    if loader is None or import_name == '__main__':
        return os.getcwd()

    # For .egg, zipimporter does not have get_filename until Python 2.7.
    # Some other loaders might exhibit the same behavior.
    if hasattr(loader, 'get_filename'):
        filepath = loader.get_filename(import_name)
    else:
        # Fall back to imports.
        __import__(import_name)
        mod = sys.modules[import_name]
        filepath = getattr(mod, '__file__', None)

        # If we don't have a filepath it might be because we are a
        # namespace package.  In this case we pick the root path from the
        # first module that is contained in our package.
        if filepath is None:
            raise RuntimeError('No root path can be found for the provided '
                               'module "%s".  This can happen because the '
                               'module came from an import hook that does '
                               'not provide file name information or because '
                               'it\'s a namespace package.  In this case '
                               'the root path needs to be explicitly '
                               'provided.' % import_name)

    # filepath is import_name.py for a module, or __init__.py for a package.
    return os.path.dirname(os.path.abspath(filepath))


def _matching_loader_thinks_module_is_package(loader, mod_name):
    """Given the loader that loaded a module and the module this function
    attempts to figure out if the given module is actually a package.
    """
    # If the loader can tell us if something is a package, we can
    # directly ask the loader.
    if hasattr(loader, 'is_package'):
        return loader.is_package(mod_name)
    # importlib's namespace loaders do not have this functionality but
    # all the modules it loads are packages, so we can take advantage of
    # this information.
    elif (loader.__class__.__module__ == '_frozen_importlib' and
          loader.__class__.__name__ == 'NamespaceLoader'):
        return True
    # Otherwise we need to fail with an error that explains what went
    # wrong.
    raise AttributeError(
        ('%s.is_package() method is missing but is required by Flask of '
         'PEP 302 import hooks.  If you do not use import hooks and '
         'you encounter this error please file a bug against Flask.') %
        loader.__class__.__name__)


def find_package(import_name):
    """Finds a package and returns the prefix (or None if the package is
    not installed) as well as the folder that contains the package or
    module as a tuple.  The package path returned is the module that would
    have to be added to the pythonpath in order to make it possible to
    import the module.  The prefix is the path below which a UNIX like
    folder structure exists (lib, share etc.).
    """
    root_mod_name = import_name.split('.')[0]
    loader = pkgutil.get_loader(root_mod_name)
    if loader is None or import_name == '__main__':
        # import name is not found, or interactive/main module
        package_path = os.getcwd()
    else:
        # For .egg, zipimporter does not have get_filename until Python 2.7.
        if hasattr(loader, 'get_filename'):
            filename = loader.get_filename(root_mod_name)
        elif hasattr(loader, 'archive'):
            # zipimporter's loader.archive points to the .egg or .zip
            # archive filename is dropped in call to dirname below.
            filename = loader.archive
        else:
            # At least one loader is missing both get_filename and archive:
            # Google App Engine's HardenedModulesHook
            #
            # Fall back to imports.
            __import__(import_name)
            filename = sys.modules[import_name].__file__
        package_path = os.path.abspath(os.path.dirname(filename))

        # In case the root module is a package we need to chop of the
        # rightmost part.  This needs to go through a helper function
        # because of python 3.3 namespace packages.
        if _matching_loader_thinks_module_is_package(
                loader, root_mod_name):
            package_path = os.path.dirname(package_path)

    site_parent, site_folder = os.path.split(package_path)
    py_prefix = os.path.abspath(sys.prefix)
    if package_path.startswith(py_prefix):
        return py_prefix, package_path
    elif site_folder.lower() == 'site-packages':
        parent, folder = os.path.split(site_parent)
        # Windows like installations
        if folder.lower() == 'lib':
            base_dir = parent
        # UNIX like installations
        elif os.path.basename(parent).lower() == 'lib':
            base_dir = os.path.dirname(parent)
        else:
            base_dir = site_parent
        return base_dir, package_path
    return None, package_path


class locked_cached_property(object):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value.  Works like the one in Werkzeug but has a lock for
    thread safety.
    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func
        self.lock = RLock()

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        with self.lock:
            value = obj.__dict__.get(self.__name__, _missing)
            if value is _missing:
                value = self.func(obj)
                obj.__dict__[self.__name__] = value
            return value


class _PackageBoundObject(object):

    def __init__(self, import_name, template_folder=None, root_path=None):
        #: The name of the package or module.  Do not change this once
        #: it was set by the constructor.
        self.import_name = import_name

        #: location of the templates.  ``None`` if templates should not be
        #: exposed.
        self.template_folder = template_folder

        if root_path is None:
            root_path = get_root_path(self.import_name)

        #: Where is the app root located?
        self.root_path = root_path

        self._static_folder = None
        self._static_url_path = None

    def _get_static_folder(self):
        if self._static_folder is not None:
            return os.path.join(self.root_path, self._static_folder)
    def _set_static_folder(self, value):
        self._static_folder = value
    static_folder = property(_get_static_folder, _set_static_folder, doc='''
    The absolute path to the configured static folder.
    ''')
    del _get_static_folder, _set_static_folder

    def _get_static_url_path(self):
        if self._static_url_path is not None:
            return self._static_url_path
        if self.static_folder is not None:
            return '/' + os.path.basename(self.static_folder)
    def _set_static_url_path(self, value):
        self._static_url_path = value
    static_url_path = property(_get_static_url_path, _set_static_url_path)
    del _get_static_url_path, _set_static_url_path

    @property
    def has_static_folder(self):
        """This is ``True`` if the package bound object's container has a
        folder for static files.

        .. versionadded:: 0.5
        """
        return self.static_folder is not None

    @locked_cached_property
    def jinja_loader(self):
        """The Jinja loader for this package bound object.

        .. versionadded:: 0.5
        """
        if self.template_folder is not None:
            return FileSystemLoader(os.path.join(self.root_path,
                                                 self.template_folder))

    def get_send_file_max_age(self, filename):
        """Provides default cache_timeout for the :func:`send_file` functions.

        By default, this function returns ``SEND_FILE_MAX_AGE_DEFAULT`` from
        the configuration of :data:`~flask.current_app`.

        Static file functions such as :func:`send_from_directory` use this
        function, and :func:`send_file` calls this function on
        :data:`~flask.current_app` when the given cache_timeout is ``None``. If a
        cache_timeout is given in :func:`send_file`, that timeout is used;
        otherwise, this method is called.

        This allows subclasses to change the behavior when sending files based
        on the filename.  For example, to set the cache timeout for .js files
        to 60 seconds::

            class MyFlask(flask.Flask):
                def get_send_file_max_age(self, name):
                    if name.lower().endswith('.js'):
                        return 60
                    return flask.Flask.get_send_file_max_age(self, name)

        .. versionadded:: 0.9
        """
        return total_seconds(current_app.send_file_max_age_default)

    def send_static_file(self, filename):
        """Function used internally to send static files from the static
        folder to the browser.

        .. versionadded:: 0.5
        """
        if not self.has_static_folder:
            raise RuntimeError('No static folder for this object')
        # Ensure get_send_file_max_age is called in all cases.
        # Here, we ensure get_send_file_max_age is called for Blueprints.
        cache_timeout = self.get_send_file_max_age(filename)
        return send_from_directory(self.static_folder, filename,
                                   cache_timeout=cache_timeout)

    def open_resource(self, resource, mode='rb'):
        """Opens a resource from the application's resource folder.  To see
        how this works, consider the following folder structure::

            /myapplication.py
            /schema.sql
            /static
                /style.css
            /templates
                /layout.html
                /index.html

        If you want to open the :file:`schema.sql` file you would do the
        following::

            with app.open_resource('schema.sql') as f:
                contents = f.read()
                do_something_with(contents)

        :param resource: the name of the resource.  To access resources within
                         subfolders use forward slashes as separator.
        :param mode: resource file opening mode, default is 'rb'.
        """
        if mode not in ('r', 'rb'):
            raise ValueError('Resources can only be opened for reading')
        return open(os.path.join(self.root_path, resource), mode)


def total_seconds(td):
    """Returns the total seconds from a timedelta object.

    :param timedelta td: the timedelta to be converted in seconds

    :returns: number of seconds
    :rtype: int
    """
    return td.days * 60 * 60 * 24 + td.seconds
