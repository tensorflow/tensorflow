# -*- coding: utf-8 -*-
r"""
    werkzeug.contrib.sessions
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This module contains some helper classes that help one to add session
    support to a python WSGI application.  For full client-side session
    storage see :mod:`~werkzeug.contrib.securecookie` which implements a
    secure, client-side session storage.


    Application Integration
    =======================

    ::

        from werkzeug.contrib.sessions import SessionMiddleware, \
             FilesystemSessionStore

        app = SessionMiddleware(app, FilesystemSessionStore())

    The current session will then appear in the WSGI environment as
    `werkzeug.session`.  However it's recommended to not use the middleware
    but the stores directly in the application.  However for very simple
    scripts a middleware for sessions could be sufficient.

    This module does not implement methods or ways to check if a session is
    expired.  That should be done by a cronjob and storage specific.  For
    example to prune unused filesystem sessions one could check the modified
    time of the files.  It sessions are stored in the database the new()
    method should add an expiration timestamp for the session.

    For better flexibility it's recommended to not use the middleware but the
    store and session object directly in the application dispatching::

        session_store = FilesystemSessionStore()

        def application(environ, start_response):
            request = Request(environ)
            sid = request.cookies.get('cookie_name')
            if sid is None:
                request.session = session_store.new()
            else:
                request.session = session_store.get(sid)
            response = get_the_response_object(request)
            if request.session.should_save:
                session_store.save(request.session)
                response.set_cookie('cookie_name', request.session.sid)
            return response(environ, start_response)

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import os
import tempfile
from os import path
from time import time
from random import random
from hashlib import sha1
from pickle import dump, load, HIGHEST_PROTOCOL

from werkzeug.datastructures import CallbackDict
from werkzeug.utils import dump_cookie, parse_cookie
from werkzeug.wsgi import ClosingIterator
from werkzeug.posixemulation import rename
from werkzeug._compat import PY2, text_type
from werkzeug.filesystem import get_filesystem_encoding


_sha1_re = re.compile(r'^[a-f0-9]{40}$')


def _urandom():
    if hasattr(os, 'urandom'):
        return os.urandom(30)
    return text_type(random()).encode('ascii')


def generate_key(salt=None):
    if salt is None:
        salt = repr(salt).encode('ascii')
    return sha1(b''.join([
        salt,
        str(time()).encode('ascii'),
        _urandom()
    ])).hexdigest()


class ModificationTrackingDict(CallbackDict):
    __slots__ = ('modified',)

    def __init__(self, *args, **kwargs):
        def on_update(self):
            self.modified = True
        self.modified = False
        CallbackDict.__init__(self, on_update=on_update)
        dict.update(self, *args, **kwargs)

    def copy(self):
        """Create a flat copy of the dict."""
        missing = object()
        result = object.__new__(self.__class__)
        for name in self.__slots__:
            val = getattr(self, name, missing)
            if val is not missing:
                setattr(result, name, val)
        return result

    def __copy__(self):
        return self.copy()


class Session(ModificationTrackingDict):

    """Subclass of a dict that keeps track of direct object changes.  Changes
    in mutable structures are not tracked, for those you have to set
    `modified` to `True` by hand.
    """
    __slots__ = ModificationTrackingDict.__slots__ + ('sid', 'new')

    def __init__(self, data, sid, new=False):
        ModificationTrackingDict.__init__(self, data)
        self.sid = sid
        self.new = new

    def __repr__(self):
        return '<%s %s%s>' % (
            self.__class__.__name__,
            dict.__repr__(self),
            self.should_save and '*' or ''
        )

    @property
    def should_save(self):
        """True if the session should be saved.

        .. versionchanged:: 0.6
           By default the session is now only saved if the session is
           modified, not if it is new like it was before.
        """
        return self.modified


class SessionStore(object):

    """Baseclass for all session stores.  The Werkzeug contrib module does not
    implement any useful stores besides the filesystem store, application
    developers are encouraged to create their own stores.

    :param session_class: The session class to use.  Defaults to
                          :class:`Session`.
    """

    def __init__(self, session_class=None):
        if session_class is None:
            session_class = Session
        self.session_class = session_class

    def is_valid_key(self, key):
        """Check if a key has the correct format."""
        return _sha1_re.match(key) is not None

    def generate_key(self, salt=None):
        """Simple function that generates a new session key."""
        return generate_key(salt)

    def new(self):
        """Generate a new session."""
        return self.session_class({}, self.generate_key(), True)

    def save(self, session):
        """Save a session."""

    def save_if_modified(self, session):
        """Save if a session class wants an update."""
        if session.should_save:
            self.save(session)

    def delete(self, session):
        """Delete a session."""

    def get(self, sid):
        """Get a session for this sid or a new session object.  This method
        has to check if the session key is valid and create a new session if
        that wasn't the case.
        """
        return self.session_class({}, sid, True)


#: used for temporary files by the filesystem session store
_fs_transaction_suffix = '.__wz_sess'


class FilesystemSessionStore(SessionStore):

    """Simple example session store that saves sessions on the filesystem.
    This store works best on POSIX systems and Windows Vista / Windows
    Server 2008 and newer.

    .. versionchanged:: 0.6
       `renew_missing` was added.  Previously this was considered `True`,
       now the default changed to `False` and it can be explicitly
       deactivated.

    :param path: the path to the folder used for storing the sessions.
                 If not provided the default temporary directory is used.
    :param filename_template: a string template used to give the session
                              a filename.  ``%s`` is replaced with the
                              session id.
    :param session_class: The session class to use.  Defaults to
                          :class:`Session`.
    :param renew_missing: set to `True` if you want the store to
                          give the user a new sid if the session was
                          not yet saved.
    """

    def __init__(self, path=None, filename_template='werkzeug_%s.sess',
                 session_class=None, renew_missing=False, mode=0o644):
        SessionStore.__init__(self, session_class)
        if path is None:
            path = tempfile.gettempdir()
        self.path = path
        if isinstance(filename_template, text_type) and PY2:
            filename_template = filename_template.encode(
                get_filesystem_encoding())
        assert not filename_template.endswith(_fs_transaction_suffix), \
            'filename templates may not end with %s' % _fs_transaction_suffix
        self.filename_template = filename_template
        self.renew_missing = renew_missing
        self.mode = mode

    def get_session_filename(self, sid):
        # out of the box, this should be a strict ASCII subset but
        # you might reconfigure the session object to have a more
        # arbitrary string.
        if isinstance(sid, text_type) and PY2:
            sid = sid.encode(get_filesystem_encoding())
        return path.join(self.path, self.filename_template % sid)

    def save(self, session):
        fn = self.get_session_filename(session.sid)
        fd, tmp = tempfile.mkstemp(suffix=_fs_transaction_suffix,
                                   dir=self.path)
        f = os.fdopen(fd, 'wb')
        try:
            dump(dict(session), f, HIGHEST_PROTOCOL)
        finally:
            f.close()
        try:
            rename(tmp, fn)
            os.chmod(fn, self.mode)
        except (IOError, OSError):
            pass

    def delete(self, session):
        fn = self.get_session_filename(session.sid)
        try:
            os.unlink(fn)
        except OSError:
            pass

    def get(self, sid):
        if not self.is_valid_key(sid):
            return self.new()
        try:
            f = open(self.get_session_filename(sid), 'rb')
        except IOError:
            if self.renew_missing:
                return self.new()
            data = {}
        else:
            try:
                try:
                    data = load(f)
                except Exception:
                    data = {}
            finally:
                f.close()
        return self.session_class(data, sid, False)

    def list(self):
        """Lists all sessions in the store.

        .. versionadded:: 0.6
        """
        before, after = self.filename_template.split('%s', 1)
        filename_re = re.compile(r'%s(.{5,})%s$' % (re.escape(before),
                                                    re.escape(after)))
        result = []
        for filename in os.listdir(self.path):
            #: this is a session that is still being saved.
            if filename.endswith(_fs_transaction_suffix):
                continue
            match = filename_re.match(filename)
            if match is not None:
                result.append(match.group(1))
        return result


class SessionMiddleware(object):

    """A simple middleware that puts the session object of a store provided
    into the WSGI environ.  It automatically sets cookies and restores
    sessions.

    However a middleware is not the preferred solution because it won't be as
    fast as sessions managed by the application itself and will put a key into
    the WSGI environment only relevant for the application which is against
    the concept of WSGI.

    The cookie parameters are the same as for the :func:`~dump_cookie`
    function just prefixed with ``cookie_``.  Additionally `max_age` is
    called `cookie_age` and not `cookie_max_age` because of backwards
    compatibility.
    """

    def __init__(self, app, store, cookie_name='session_id',
                 cookie_age=None, cookie_expires=None, cookie_path='/',
                 cookie_domain=None, cookie_secure=None,
                 cookie_httponly=False, environ_key='werkzeug.session'):
        self.app = app
        self.store = store
        self.cookie_name = cookie_name
        self.cookie_age = cookie_age
        self.cookie_expires = cookie_expires
        self.cookie_path = cookie_path
        self.cookie_domain = cookie_domain
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.environ_key = environ_key

    def __call__(self, environ, start_response):
        cookie = parse_cookie(environ.get('HTTP_COOKIE', ''))
        sid = cookie.get(self.cookie_name, None)
        if sid is None:
            session = self.store.new()
        else:
            session = self.store.get(sid)
        environ[self.environ_key] = session

        def injecting_start_response(status, headers, exc_info=None):
            if session.should_save:
                self.store.save(session)
                headers.append(('Set-Cookie', dump_cookie(self.cookie_name,
                                                          session.sid, self.cookie_age,
                                                          self.cookie_expires, self.cookie_path,
                                                          self.cookie_domain, self.cookie_secure,
                                                          self.cookie_httponly)))
            return start_response(status, headers, exc_info)
        return ClosingIterator(self.app(environ, injecting_start_response),
                               lambda: self.store.save_if_modified(session))
