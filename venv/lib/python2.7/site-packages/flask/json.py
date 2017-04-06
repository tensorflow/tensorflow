# -*- coding: utf-8 -*-
"""
    flask.jsonimpl
    ~~~~~~~~~~~~~~

    Implementation helpers for the JSON support in Flask.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
import io
import uuid
from datetime import date
from .globals import current_app, request
from ._compat import text_type, PY2

from werkzeug.http import http_date
from jinja2 import Markup

# Use the same json implementation as itsdangerous on which we
# depend anyways.
from itsdangerous import json as _json


# Figure out if simplejson escapes slashes.  This behavior was changed
# from one version to another without reason.
_slash_escape = '\\/' not in _json.dumps('/')


__all__ = ['dump', 'dumps', 'load', 'loads', 'htmlsafe_dump',
           'htmlsafe_dumps', 'JSONDecoder', 'JSONEncoder',
           'jsonify']


def _wrap_reader_for_text(fp, encoding):
    if isinstance(fp.read(0), bytes):
        fp = io.TextIOWrapper(io.BufferedReader(fp), encoding)
    return fp


def _wrap_writer_for_text(fp, encoding):
    try:
        fp.write('')
    except TypeError:
        fp = io.TextIOWrapper(fp, encoding)
    return fp


class JSONEncoder(_json.JSONEncoder):
    """The default Flask JSON encoder.  This one extends the default simplejson
    encoder by also supporting ``datetime`` objects, ``UUID`` as well as
    ``Markup`` objects which are serialized as RFC 822 datetime strings (same
    as the HTTP date format).  In order to support more data types override the
    :meth:`default` method.
    """

    def default(self, o):
        """Implement this method in a subclass such that it returns a
        serializable object for ``o``, or calls the base implementation (to
        raise a :exc:`TypeError`).

        For example, to support arbitrary iterators, you could implement
        default like this::

            def default(self, o):
                try:
                    iterable = iter(o)
                except TypeError:
                    pass
                else:
                    return list(iterable)
                return JSONEncoder.default(self, o)
        """
        if isinstance(o, date):
            return http_date(o.timetuple())
        if isinstance(o, uuid.UUID):
            return str(o)
        if hasattr(o, '__html__'):
            return text_type(o.__html__())
        return _json.JSONEncoder.default(self, o)


class JSONDecoder(_json.JSONDecoder):
    """The default JSON decoder.  This one does not change the behavior from
    the default simplejson decoder.  Consult the :mod:`json` documentation
    for more information.  This decoder is not only used for the load
    functions of this module but also :attr:`~flask.Request`.
    """


def _dump_arg_defaults(kwargs):
    """Inject default arguments for dump functions."""
    if current_app:
        kwargs.setdefault('cls', current_app.json_encoder)
        if not current_app.config['JSON_AS_ASCII']:
            kwargs.setdefault('ensure_ascii', False)
        kwargs.setdefault('sort_keys', current_app.config['JSON_SORT_KEYS'])
    else:
        kwargs.setdefault('sort_keys', True)
        kwargs.setdefault('cls', JSONEncoder)


def _load_arg_defaults(kwargs):
    """Inject default arguments for load functions."""
    if current_app:
        kwargs.setdefault('cls', current_app.json_decoder)
    else:
        kwargs.setdefault('cls', JSONDecoder)


def dumps(obj, **kwargs):
    """Serialize ``obj`` to a JSON formatted ``str`` by using the application's
    configured encoder (:attr:`~flask.Flask.json_encoder`) if there is an
    application on the stack.

    This function can return ``unicode`` strings or ascii-only bytestrings by
    default which coerce into unicode strings automatically.  That behavior by
    default is controlled by the ``JSON_AS_ASCII`` configuration variable
    and can be overridden by the simplejson ``ensure_ascii`` parameter.
    """
    _dump_arg_defaults(kwargs)
    encoding = kwargs.pop('encoding', None)
    rv = _json.dumps(obj, **kwargs)
    if encoding is not None and isinstance(rv, text_type):
        rv = rv.encode(encoding)
    return rv


def dump(obj, fp, **kwargs):
    """Like :func:`dumps` but writes into a file object."""
    _dump_arg_defaults(kwargs)
    encoding = kwargs.pop('encoding', None)
    if encoding is not None:
        fp = _wrap_writer_for_text(fp, encoding)
    _json.dump(obj, fp, **kwargs)


def loads(s, **kwargs):
    """Unserialize a JSON object from a string ``s`` by using the application's
    configured decoder (:attr:`~flask.Flask.json_decoder`) if there is an
    application on the stack.
    """
    _load_arg_defaults(kwargs)
    if isinstance(s, bytes):
        s = s.decode(kwargs.pop('encoding', None) or 'utf-8')
    return _json.loads(s, **kwargs)


def load(fp, **kwargs):
    """Like :func:`loads` but reads from a file object.
    """
    _load_arg_defaults(kwargs)
    if not PY2:
        fp = _wrap_reader_for_text(fp, kwargs.pop('encoding', None) or 'utf-8')
    return _json.load(fp, **kwargs)


def htmlsafe_dumps(obj, **kwargs):
    """Works exactly like :func:`dumps` but is safe for use in ``<script>``
    tags.  It accepts the same arguments and returns a JSON string.  Note that
    this is available in templates through the ``|tojson`` filter which will
    also mark the result as safe.  Due to how this function escapes certain
    characters this is safe even if used outside of ``<script>`` tags.

    The following characters are escaped in strings:

    -   ``<``
    -   ``>``
    -   ``&``
    -   ``'``

    This makes it safe to embed such strings in any place in HTML with the
    notable exception of double quoted attributes.  In that case single
    quote your attributes or HTML escape it in addition.

    .. versionchanged:: 0.10
       This function's return value is now always safe for HTML usage, even
       if outside of script tags or if used in XHTML.  This rule does not
       hold true when using this function in HTML attributes that are double
       quoted.  Always single quote attributes if you use the ``|tojson``
       filter.  Alternatively use ``|tojson|forceescape``.
    """
    rv = dumps(obj, **kwargs) \
        .replace(u'<', u'\\u003c') \
        .replace(u'>', u'\\u003e') \
        .replace(u'&', u'\\u0026') \
        .replace(u"'", u'\\u0027')
    if not _slash_escape:
        rv = rv.replace('\\/', '/')
    return rv


def htmlsafe_dump(obj, fp, **kwargs):
    """Like :func:`htmlsafe_dumps` but writes into a file object."""
    fp.write(text_type(htmlsafe_dumps(obj, **kwargs)))


def jsonify(*args, **kwargs):
    """This function wraps :func:`dumps` to add a few enhancements that make
    life easier.  It turns the JSON output into a :class:`~flask.Response`
    object with the :mimetype:`application/json` mimetype.  For convenience, it
    also converts multiple arguments into an array or multiple keyword arguments
    into a dict.  This means that both ``jsonify(1,2,3)`` and
    ``jsonify([1,2,3])`` serialize to ``[1,2,3]``.

    For clarity, the JSON serialization behavior has the following differences
    from :func:`dumps`:

    1. Single argument: Passed straight through to :func:`dumps`.
    2. Multiple arguments: Converted to an array before being passed to
       :func:`dumps`.
    3. Multiple keyword arguments: Converted to a dict before being passed to
       :func:`dumps`.
    4. Both args and kwargs: Behavior undefined and will throw an exception.

    Example usage::

        from flask import jsonify

        @app.route('/_get_current_user')
        def get_current_user():
            return jsonify(username=g.user.username,
                           email=g.user.email,
                           id=g.user.id)

    This will send a JSON response like this to the browser::

        {
            "username": "admin",
            "email": "admin@localhost",
            "id": 42
        }


    .. versionchanged:: 0.11
       Added support for serializing top-level arrays. This introduces a
       security risk in ancient browsers. See :ref:`json-security` for details.

    This function's response will be pretty printed if it was not requested
    with ``X-Requested-With: XMLHttpRequest`` to simplify debugging unless
    the ``JSONIFY_PRETTYPRINT_REGULAR`` config parameter is set to false.
    Compressed (not pretty) formatting currently means no indents and no
    spaces after separators.

    .. versionadded:: 0.2
    """

    indent = None
    separators = (',', ':')

    if current_app.config['JSONIFY_PRETTYPRINT_REGULAR'] and not request.is_xhr:
        indent = 2
        separators = (', ', ': ')

    if args and kwargs:
        raise TypeError('jsonify() behavior undefined when passed both args and kwargs')
    elif len(args) == 1:  # single args are passed directly to dumps()
        data = args[0]
    else:
        data = args or kwargs

    return current_app.response_class(
        (dumps(data, indent=indent, separators=separators), '\n'),
        mimetype=current_app.config['JSONIFY_MIMETYPE']
    )


def tojson_filter(obj, **kwargs):
    return Markup(htmlsafe_dumps(obj, **kwargs))
