# -*- coding: utf-8 -*-
r"""
    werkzeug.contrib.securecookie
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module implements a cookie that is not alterable from the client
    because it adds a checksum the server checks for.  You can use it as
    session replacement if all you have is a user id or something to mark
    a logged in user.

    Keep in mind that the data is still readable from the client as a
    normal cookie is.  However you don't have to store and flush the
    sessions you have at the server.

    Example usage:

    >>> from werkzeug.contrib.securecookie import SecureCookie
    >>> x = SecureCookie({"foo": 42, "baz": (1, 2, 3)}, "deadbeef")

    Dumping into a string so that one can store it in a cookie:

    >>> value = x.serialize()

    Loading from that string again:

    >>> x = SecureCookie.unserialize(value, "deadbeef")
    >>> x["baz"]
    (1, 2, 3)

    If someone modifies the cookie and the checksum is wrong the unserialize
    method will fail silently and return a new empty `SecureCookie` object.

    Keep in mind that the values will be visible in the cookie so do not
    store data in a cookie you don't want the user to see.

    Application Integration
    =======================

    If you are using the werkzeug request objects you could integrate the
    secure cookie into your application like this::

        from werkzeug.utils import cached_property
        from werkzeug.wrappers import BaseRequest
        from werkzeug.contrib.securecookie import SecureCookie

        # don't use this key but a different one; you could just use
        # os.urandom(20) to get something random
        SECRET_KEY = '\xfa\xdd\xb8z\xae\xe0}4\x8b\xea'

        class Request(BaseRequest):

            @cached_property
            def client_session(self):
                data = self.cookies.get('session_data')
                if not data:
                    return SecureCookie(secret_key=SECRET_KEY)
                return SecureCookie.unserialize(data, SECRET_KEY)

        def application(environ, start_response):
            request = Request(environ, start_response)

            # get a response object here
            response = ...

            if request.client_session.should_save:
                session_data = request.client_session.serialize()
                response.set_cookie('session_data', session_data,
                                    httponly=True)
            return response(environ, start_response)

    A less verbose integration can be achieved by using shorthand methods::

        class Request(BaseRequest):

            @cached_property
            def client_session(self):
                return SecureCookie.load_cookie(self, secret_key=COOKIE_SECRET)

        def application(environ, start_response):
            request = Request(environ, start_response)

            # get a response object here
            response = ...

            request.client_session.save_cookie(response)
            return response(environ, start_response)

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import pickle
import base64
from hmac import new as hmac
from time import time
from hashlib import sha1 as _default_hash

from werkzeug._compat import iteritems, text_type
from werkzeug.urls import url_quote_plus, url_unquote_plus
from werkzeug._internal import _date_to_unix
from werkzeug.contrib.sessions import ModificationTrackingDict
from werkzeug.security import safe_str_cmp
from werkzeug._compat import to_native


class UnquoteError(Exception):

    """Internal exception used to signal failures on quoting."""


class SecureCookie(ModificationTrackingDict):

    """Represents a secure cookie.  You can subclass this class and provide
    an alternative mac method.  The import thing is that the mac method
    is a function with a similar interface to the hashlib.  Required
    methods are update() and digest().

    Example usage:

    >>> x = SecureCookie({"foo": 42, "baz": (1, 2, 3)}, "deadbeef")
    >>> x["foo"]
    42
    >>> x["baz"]
    (1, 2, 3)
    >>> x["blafasel"] = 23
    >>> x.should_save
    True

    :param data: the initial data.  Either a dict, list of tuples or `None`.
    :param secret_key: the secret key.  If not set `None` or not specified
                       it has to be set before :meth:`serialize` is called.
    :param new: The initial value of the `new` flag.
    """

    #: The hash method to use.  This has to be a module with a new function
    #: or a function that creates a hashlib object.  Such as `hashlib.md5`
    #: Subclasses can override this attribute.  The default hash is sha1.
    #: Make sure to wrap this in staticmethod() if you store an arbitrary
    #: function there such as hashlib.sha1 which  might be implemented
    #: as a function.
    hash_method = staticmethod(_default_hash)

    #: the module used for serialization.  Unless overriden by subclasses
    #: the standard pickle module is used.
    serialization_method = pickle

    #: if the contents should be base64 quoted.  This can be disabled if the
    #: serialization process returns cookie safe strings only.
    quote_base64 = True

    def __init__(self, data=None, secret_key=None, new=True):
        ModificationTrackingDict.__init__(self, data or ())
        # explicitly convert it into a bytestring because python 2.6
        # no longer performs an implicit string conversion on hmac
        if secret_key is not None:
            secret_key = bytes(secret_key)
        self.secret_key = secret_key
        self.new = new

    def __repr__(self):
        return '<%s %s%s>' % (
            self.__class__.__name__,
            dict.__repr__(self),
            self.should_save and '*' or ''
        )

    @property
    def should_save(self):
        """True if the session should be saved.  By default this is only true
        for :attr:`modified` cookies, not :attr:`new`.
        """
        return self.modified

    @classmethod
    def quote(cls, value):
        """Quote the value for the cookie.  This can be any object supported
        by :attr:`serialization_method`.

        :param value: the value to quote.
        """
        if cls.serialization_method is not None:
            value = cls.serialization_method.dumps(value)
        if cls.quote_base64:
            value = b''.join(base64.b64encode(value).splitlines()).strip()
        return value

    @classmethod
    def unquote(cls, value):
        """Unquote the value for the cookie.  If unquoting does not work a
        :exc:`UnquoteError` is raised.

        :param value: the value to unquote.
        """
        try:
            if cls.quote_base64:
                value = base64.b64decode(value)
            if cls.serialization_method is not None:
                value = cls.serialization_method.loads(value)
            return value
        except Exception:
            # unfortunately pickle and other serialization modules can
            # cause pretty every error here.  if we get one we catch it
            # and convert it into an UnquoteError
            raise UnquoteError()

    def serialize(self, expires=None):
        """Serialize the secure cookie into a string.

        If expires is provided, the session will be automatically invalidated
        after expiration when you unseralize it. This provides better
        protection against session cookie theft.

        :param expires: an optional expiration date for the cookie (a
                        :class:`datetime.datetime` object)
        """
        if self.secret_key is None:
            raise RuntimeError('no secret key defined')
        if expires:
            self['_expires'] = _date_to_unix(expires)
        result = []
        mac = hmac(self.secret_key, None, self.hash_method)
        for key, value in sorted(self.items()):
            result.append(('%s=%s' % (
                url_quote_plus(key),
                self.quote(value).decode('ascii')
            )).encode('ascii'))
            mac.update(b'|' + result[-1])
        return b'?'.join([
            base64.b64encode(mac.digest()).strip(),
            b'&'.join(result)
        ])

    @classmethod
    def unserialize(cls, string, secret_key):
        """Load the secure cookie from a serialized string.

        :param string: the cookie value to unserialize.
        :param secret_key: the secret key used to serialize the cookie.
        :return: a new :class:`SecureCookie`.
        """
        if isinstance(string, text_type):
            string = string.encode('utf-8', 'replace')
        if isinstance(secret_key, text_type):
            secret_key = secret_key.encode('utf-8', 'replace')
        try:
            base64_hash, data = string.split(b'?', 1)
        except (ValueError, IndexError):
            items = ()
        else:
            items = {}
            mac = hmac(secret_key, None, cls.hash_method)
            for item in data.split(b'&'):
                mac.update(b'|' + item)
                if b'=' not in item:
                    items = None
                    break
                key, value = item.split(b'=', 1)
                # try to make the key a string
                key = url_unquote_plus(key.decode('ascii'))
                try:
                    key = to_native(key)
                except UnicodeError:
                    pass
                items[key] = value

            # no parsing error and the mac looks okay, we can now
            # sercurely unpickle our cookie.
            try:
                client_hash = base64.b64decode(base64_hash)
            except TypeError:
                items = client_hash = None
            if items is not None and safe_str_cmp(client_hash, mac.digest()):
                try:
                    for key, value in iteritems(items):
                        items[key] = cls.unquote(value)
                except UnquoteError:
                    items = ()
                else:
                    if '_expires' in items:
                        if time() > items['_expires']:
                            items = ()
                        else:
                            del items['_expires']
            else:
                items = ()
        return cls(items, secret_key, False)

    @classmethod
    def load_cookie(cls, request, key='session', secret_key=None):
        """Loads a :class:`SecureCookie` from a cookie in request.  If the
        cookie is not set, a new :class:`SecureCookie` instanced is
        returned.

        :param request: a request object that has a `cookies` attribute
                        which is a dict of all cookie values.
        :param key: the name of the cookie.
        :param secret_key: the secret key used to unquote the cookie.
                           Always provide the value even though it has
                           no default!
        """
        data = request.cookies.get(key)
        if not data:
            return cls(secret_key=secret_key)
        return cls.unserialize(data, secret_key)

    def save_cookie(self, response, key='session', expires=None,
                    session_expires=None, max_age=None, path='/', domain=None,
                    secure=None, httponly=False, force=False):
        """Saves the SecureCookie in a cookie on response object.  All
        parameters that are not described here are forwarded directly
        to :meth:`~BaseResponse.set_cookie`.

        :param response: a response object that has a
                         :meth:`~BaseResponse.set_cookie` method.
        :param key: the name of the cookie.
        :param session_expires: the expiration date of the secure cookie
                                stored information.  If this is not provided
                                the cookie `expires` date is used instead.
        """
        if force or self.should_save:
            data = self.serialize(session_expires or expires)
            response.set_cookie(key, data, expires=expires, max_age=max_age,
                                path=path, domain=domain, secure=secure,
                                httponly=httponly)
