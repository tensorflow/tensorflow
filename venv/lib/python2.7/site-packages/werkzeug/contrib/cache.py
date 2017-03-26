# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.cache
    ~~~~~~~~~~~~~~~~~~~~~~

    The main problem with dynamic Web sites is, well, they're dynamic.  Each
    time a user requests a page, the webserver executes a lot of code, queries
    the database, renders templates until the visitor gets the page he sees.

    This is a lot more expensive than just loading a file from the file system
    and sending it to the visitor.

    For most Web applications, this overhead isn't a big deal but once it
    becomes, you will be glad to have a cache system in place.

    How Caching Works
    =================

    Caching is pretty simple.  Basically you have a cache object lurking around
    somewhere that is connected to a remote cache or the file system or
    something else.  When the request comes in you check if the current page
    is already in the cache and if so, you're returning it from the cache.
    Otherwise you generate the page and put it into the cache. (Or a fragment
    of the page, you don't have to cache the full thing)

    Here is a simple example of how to cache a sidebar for a template::

        def get_sidebar(user):
            identifier = 'sidebar_for/user%d' % user.id
            value = cache.get(identifier)
            if value is not None:
                return value
            value = generate_sidebar_for(user=user)
            cache.set(identifier, value, timeout=60 * 5)
            return value

    Creating a Cache Object
    =======================

    To create a cache object you just import the cache system of your choice
    from the cache module and instantiate it.  Then you can start working
    with that object:

    >>> from werkzeug.contrib.cache import SimpleCache
    >>> c = SimpleCache()
    >>> c.set("foo", "value")
    >>> c.get("foo")
    'value'
    >>> c.get("missing") is None
    True

    Please keep in mind that you have to create the cache and put it somewhere
    you have access to it (either as a module global you can import or you just
    put it into your WSGI application).

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import os
import re
import errno
import tempfile
from hashlib import md5
from time import time
try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle

from werkzeug._compat import iteritems, string_types, text_type, \
    integer_types, to_native
from werkzeug.posixemulation import rename


def _items(mappingorseq):
    """Wrapper for efficient iteration over mappings represented by dicts
    or sequences::

        >>> for k, v in _items((i, i*i) for i in xrange(5)):
        ...    assert k*k == v

        >>> for k, v in _items(dict((i, i*i) for i in xrange(5))):
        ...    assert k*k == v

    """
    if hasattr(mappingorseq, 'items'):
        return iteritems(mappingorseq)
    return mappingorseq


class BaseCache(object):

    """Baseclass for the cache systems.  All the cache systems implement this
    API or a superset of it.

    :param default_timeout: the default timeout (in seconds) that is used if no
                            timeout is specified on :meth:`set`. A timeout of 0
                            indicates that the cache never expires.
    """

    def __init__(self, default_timeout=300):
        self.default_timeout = default_timeout

    def get(self, key):
        """Look up key in the cache and return the value for it.

        :param key: the key to be looked up.
        :returns: The value if it exists and is readable, else ``None``.
        """
        return None

    def delete(self, key):
        """Delete `key` from the cache.

        :param key: the key to delete.
        :returns: Whether the key existed and has been deleted.
        :rtype: boolean
        """
        return True

    def get_many(self, *keys):
        """Returns a list of values for the given keys.
        For each key a item in the list is created::

            foo, bar = cache.get_many("foo", "bar")

        Has the same error handling as :meth:`get`.

        :param keys: The function accepts multiple keys as positional
                     arguments.
        """
        return map(self.get, keys)

    def get_dict(self, *keys):
        """Like :meth:`get_many` but return a dict::

            d = cache.get_dict("foo", "bar")
            foo = d["foo"]
            bar = d["bar"]

        :param keys: The function accepts multiple keys as positional
                     arguments.
        """
        return dict(zip(keys, self.get_many(*keys)))

    def set(self, key, value, timeout=None):
        """Add a new key/value to the cache (overwrites value, if key already
        exists in the cache).

        :param key: the key to set
        :param value: the value for the key
        :param timeout: the cache timeout for the key (if not specified,
                        it uses the default timeout). A timeout of 0 idicates
                        that the cache never expires.
        :returns: ``True`` if key has been updated, ``False`` for backend
                  errors. Pickling errors, however, will raise a subclass of
                  ``pickle.PickleError``.
        :rtype: boolean
        """
        return True

    def add(self, key, value, timeout=None):
        """Works like :meth:`set` but does not overwrite the values of already
        existing keys.

        :param key: the key to set
        :param value: the value for the key
        :param timeout: the cache timeout for the key or the default
                        timeout if not specified. A timeout of 0 indicates
                        that the cache never expires.
        :returns: Same as :meth:`set`, but also ``False`` for already
                  existing keys.
        :rtype: boolean
        """
        return True

    def set_many(self, mapping, timeout=None):
        """Sets multiple keys and values from a mapping.

        :param mapping: a mapping with the keys/values to set.
        :param timeout: the cache timeout for the key (if not specified,
                        it uses the default timeout). A timeout of 0
                        indicates tht the cache never expires.
        :returns: Whether all given keys have been set.
        :rtype: boolean
        """
        rv = True
        for key, value in _items(mapping):
            if not self.set(key, value, timeout):
                rv = False
        return rv

    def delete_many(self, *keys):
        """Deletes multiple keys at once.

        :param keys: The function accepts multiple keys as positional
                     arguments.
        :returns: Whether all given keys have been deleted.
        :rtype: boolean
        """
        return all(self.delete(key) for key in keys)

    def has(self, key):
        """Checks if a key exists in the cache without returning it. This is a
        cheap operation that bypasses loading the actual data on the backend.

        This method is optional and may not be implemented on all caches.

        :param key: the key to check
        """
        raise NotImplementedError(
            '%s doesn\'t have an efficient implementation of `has`. That '
            'means it is impossible to check whether a key exists without '
            'fully loading the key\'s data. Consider using `self.get` '
            'explicitly if you don\'t care about performance.'
        )

    def clear(self):
        """Clears the cache.  Keep in mind that not all caches support
        completely clearing the cache.
        :returns: Whether the cache has been cleared.
        :rtype: boolean
        """
        return True

    def inc(self, key, delta=1):
        """Increments the value of a key by `delta`.  If the key does
        not yet exist it is initialized with `delta`.

        For supporting caches this is an atomic operation.

        :param key: the key to increment.
        :param delta: the delta to add.
        :returns: The new value or ``None`` for backend errors.
        """
        value = (self.get(key) or 0) + delta
        return value if self.set(key, value) else None

    def dec(self, key, delta=1):
        """Decrements the value of a key by `delta`.  If the key does
        not yet exist it is initialized with `-delta`.

        For supporting caches this is an atomic operation.

        :param key: the key to increment.
        :param delta: the delta to subtract.
        :returns: The new value or `None` for backend errors.
        """
        value = (self.get(key) or 0) - delta
        return value if self.set(key, value) else None


class NullCache(BaseCache):

    """A cache that doesn't cache.  This can be useful for unit testing.

    :param default_timeout: a dummy parameter that is ignored but exists
                            for API compatibility with other caches.
    """


class SimpleCache(BaseCache):

    """Simple memory cache for single process environments.  This class exists
    mainly for the development server and is not 100% thread safe.  It tries
    to use as many atomic operations as possible and no locks for simplicity
    but it could happen under heavy load that keys are added multiple times.

    :param threshold: the maximum number of items the cache stores before
                      it starts deleting some.
    :param default_timeout: the default timeout that is used if no timeout is
                            specified on :meth:`~BaseCache.set`. A timeout of
                            0 indicates that the cache never expires.
    """

    def __init__(self, threshold=500, default_timeout=300):
        BaseCache.__init__(self, default_timeout)
        self._cache = {}
        self.clear = self._cache.clear
        self._threshold = threshold

    def _prune(self):
        if len(self._cache) > self._threshold:
            now = time()
            toremove = []
            for idx, (key, (expires, _)) in enumerate(self._cache.items()):
                if (expires != 0 and expires <= now) or idx % 3 == 0:
                    toremove.append(key)
            for key in toremove:
                self._cache.pop(key, None)

    def _get_expiration(self, timeout):
        if timeout is None:
            timeout = self.default_timeout
        if timeout > 0:
            timeout = time() + timeout
        return timeout

    def get(self, key):
        try:
            expires, value = self._cache[key]
            if expires == 0 or expires > time():
                return pickle.loads(value)
        except (KeyError, pickle.PickleError):
            return None

    def set(self, key, value, timeout=None):
        expires = self._get_expiration(timeout)
        self._prune()
        self._cache[key] = (expires, pickle.dumps(value,
                                                  pickle.HIGHEST_PROTOCOL))
        return True

    def add(self, key, value, timeout=None):
        expires = self._get_expiration(timeout)
        self._prune()
        item = (expires, pickle.dumps(value,
                                      pickle.HIGHEST_PROTOCOL))
        if key in self._cache:
            return False
        self._cache.setdefault(key, item)
        return True

    def delete(self, key):
        return self._cache.pop(key, None) is not None

    def has(self, key):
        try:
            expires, value = self._cache[key]
            return expires == 0 or expires > time()
        except KeyError:
            return False

_test_memcached_key = re.compile(r'[^\x00-\x21\xff]{1,250}$').match


class MemcachedCache(BaseCache):

    """A cache that uses memcached as backend.

    The first argument can either be an object that resembles the API of a
    :class:`memcache.Client` or a tuple/list of server addresses. In the
    event that a tuple/list is passed, Werkzeug tries to import the best
    available memcache library.

    This cache looks into the following packages/modules to find bindings for
    memcached:

        - ``pylibmc``
        - ``google.appengine.api.memcached``
        - ``memcached``

    Implementation notes:  This cache backend works around some limitations in
    memcached to simplify the interface.  For example unicode keys are encoded
    to utf-8 on the fly.  Methods such as :meth:`~BaseCache.get_dict` return
    the keys in the same format as passed.  Furthermore all get methods
    silently ignore key errors to not cause problems when untrusted user data
    is passed to the get methods which is often the case in web applications.

    :param servers: a list or tuple of server addresses or alternatively
                    a :class:`memcache.Client` or a compatible client.
    :param default_timeout: the default timeout that is used if no timeout is
                            specified on :meth:`~BaseCache.set`. A timeout of
                            0 indicates taht the cache never expires.
    :param key_prefix: a prefix that is added before all keys.  This makes it
                       possible to use the same memcached server for different
                       applications.  Keep in mind that
                       :meth:`~BaseCache.clear` will also clear keys with a
                       different prefix.
    """

    def __init__(self, servers=None, default_timeout=300, key_prefix=None):
        BaseCache.__init__(self, default_timeout)
        if servers is None or isinstance(servers, (list, tuple)):
            if servers is None:
                servers = ['127.0.0.1:11211']
            self._client = self.import_preferred_memcache_lib(servers)
            if self._client is None:
                raise RuntimeError('no memcache module found')
        else:
            # NOTE: servers is actually an already initialized memcache
            # client.
            self._client = servers

        self.key_prefix = to_native(key_prefix)

    def _normalize_key(self, key):
        key = to_native(key, 'utf-8')
        if self.key_prefix:
            key = self.key_prefix + key
        return key

    def _normalize_timeout(self, timeout):
        if timeout is None:
            timeout = self.default_timeout
        if timeout > 0:
            timeout = int(time()) + timeout
        return timeout

    def get(self, key):
        key = self._normalize_key(key)
        # memcached doesn't support keys longer than that.  Because often
        # checks for so long keys can occur because it's tested from user
        # submitted data etc we fail silently for getting.
        if _test_memcached_key(key):
            return self._client.get(key)

    def get_dict(self, *keys):
        key_mapping = {}
        have_encoded_keys = False
        for key in keys:
            encoded_key = self._normalize_key(key)
            if not isinstance(key, str):
                have_encoded_keys = True
            if _test_memcached_key(key):
                key_mapping[encoded_key] = key
        d = rv = self._client.get_multi(key_mapping.keys())
        if have_encoded_keys or self.key_prefix:
            rv = {}
            for key, value in iteritems(d):
                rv[key_mapping[key]] = value
        if len(rv) < len(keys):
            for key in keys:
                if key not in rv:
                    rv[key] = None
        return rv

    def add(self, key, value, timeout=None):
        key = self._normalize_key(key)
        timeout = self._normalize_timeout(timeout)
        return self._client.add(key, value, timeout)

    def set(self, key, value, timeout=None):
        key = self._normalize_key(key)
        timeout = self._normalize_timeout(timeout)
        return self._client.set(key, value, timeout)

    def get_many(self, *keys):
        d = self.get_dict(*keys)
        return [d[key] for key in keys]

    def set_many(self, mapping, timeout=None):
        new_mapping = {}
        for key, value in _items(mapping):
            key = self._normalize_key(key)
            new_mapping[key] = value

        timeout = self._normalize_timeout(timeout)
        failed_keys = self._client.set_multi(new_mapping, timeout)
        return not failed_keys

    def delete(self, key):
        key = self._normalize_key(key)
        if _test_memcached_key(key):
            return self._client.delete(key)

    def delete_many(self, *keys):
        new_keys = []
        for key in keys:
            key = self._normalize_key(key)
            if _test_memcached_key(key):
                new_keys.append(key)
        return self._client.delete_multi(new_keys)

    def has(self, key):
        key = self._normalize_key(key)
        if _test_memcached_key(key):
            return self._client.append(key, '')
        return False

    def clear(self):
        return self._client.flush_all()

    def inc(self, key, delta=1):
        key = self._normalize_key(key)
        return self._client.incr(key, delta)

    def dec(self, key, delta=1):
        key = self._normalize_key(key)
        return self._client.decr(key, delta)

    def import_preferred_memcache_lib(self, servers):
        """Returns an initialized memcache client.  Used by the constructor."""
        try:
            import pylibmc
        except ImportError:
            pass
        else:
            return pylibmc.Client(servers)

        try:
            from google.appengine.api import memcache
        except ImportError:
            pass
        else:
            return memcache.Client()

        try:
            import memcache
        except ImportError:
            pass
        else:
            return memcache.Client(servers)


# backwards compatibility
GAEMemcachedCache = MemcachedCache


class RedisCache(BaseCache):

    """Uses the Redis key-value store as a cache backend.

    The first argument can be either a string denoting address of the Redis
    server or an object resembling an instance of a redis.Redis class.

    Note: Python Redis API already takes care of encoding unicode strings on
    the fly.

    .. versionadded:: 0.7

    .. versionadded:: 0.8
       `key_prefix` was added.

    .. versionchanged:: 0.8
       This cache backend now properly serializes objects.

    .. versionchanged:: 0.8.3
       This cache backend now supports password authentication.

    .. versionchanged:: 0.10
        ``**kwargs`` is now passed to the redis object.

    :param host: address of the Redis server or an object which API is
                 compatible with the official Python Redis client (redis-py).
    :param port: port number on which Redis server listens for connections.
    :param password: password authentication for the Redis server.
    :param db: db (zero-based numeric index) on Redis Server to connect.
    :param default_timeout: the default timeout that is used if no timeout is
                            specified on :meth:`~BaseCache.set`. A timeout of
                            0 indicates that the cache never expires.
    :param key_prefix: A prefix that should be added to all keys.

    Any additional keyword arguments will be passed to ``redis.Redis``.
    """

    def __init__(self, host='localhost', port=6379, password=None,
                 db=0, default_timeout=300, key_prefix=None, **kwargs):
        BaseCache.__init__(self, default_timeout)
        if isinstance(host, string_types):
            try:
                import redis
            except ImportError:
                raise RuntimeError('no redis module found')
            if kwargs.get('decode_responses', None):
                raise ValueError('decode_responses is not supported by '
                                 'RedisCache.')
            self._client = redis.Redis(host=host, port=port, password=password,
                                       db=db, **kwargs)
        else:
            self._client = host
        self.key_prefix = key_prefix or ''

    def _get_expiration(self, timeout):
        if timeout is None:
            timeout = self.default_timeout
        if timeout == 0:
            timeout = -1
        return timeout

    def dump_object(self, value):
        """Dumps an object into a string for redis.  By default it serializes
        integers as regular string and pickle dumps everything else.
        """
        t = type(value)
        if t in integer_types:
            return str(value).encode('ascii')
        return b'!' + pickle.dumps(value)

    def load_object(self, value):
        """The reversal of :meth:`dump_object`.  This might be called with
        None.
        """
        if value is None:
            return None
        if value.startswith(b'!'):
            try:
                return pickle.loads(value[1:])
            except pickle.PickleError:
                return None
        try:
            return int(value)
        except ValueError:
            # before 0.8 we did not have serialization.  Still support that.
            return value

    def get(self, key):
        return self.load_object(self._client.get(self.key_prefix + key))

    def get_many(self, *keys):
        if self.key_prefix:
            keys = [self.key_prefix + key for key in keys]
        return [self.load_object(x) for x in self._client.mget(keys)]

    def set(self, key, value, timeout=None):
        timeout = self._get_expiration(timeout)
        dump = self.dump_object(value)
        if timeout == -1:
            result = self._client.set(name=self.key_prefix + key,
                                      value=dump)
        else:
            result = self._client.setex(name=self.key_prefix + key,
                                        value=dump, time=timeout)
        return result

    def add(self, key, value, timeout=None):
        timeout = self._get_expiration(timeout)
        dump = self.dump_object(value)
        return (
            self._client.setnx(name=self.key_prefix + key, value=dump) and
            self._client.expire(name=self.key_prefix + key, time=timeout)
        )

    def set_many(self, mapping, timeout=None):
        timeout = self._get_expiration(timeout)
        # Use transaction=False to batch without calling redis MULTI
        # which is not supported by twemproxy
        pipe = self._client.pipeline(transaction=False)

        for key, value in _items(mapping):
            dump = self.dump_object(value)
            if timeout == -1:
                pipe.set(name=self.key_prefix + key, value=dump)
            else:
                pipe.setex(name=self.key_prefix + key, value=dump,
                           time=timeout)
        return pipe.execute()

    def delete(self, key):
        return self._client.delete(self.key_prefix + key)

    def delete_many(self, *keys):
        if not keys:
            return
        if self.key_prefix:
            keys = [self.key_prefix + key for key in keys]
        return self._client.delete(*keys)

    def has(self, key):
        return self._client.exists(self.key_prefix + key)

    def clear(self):
        status = False
        if self.key_prefix:
            keys = self._client.keys(self.key_prefix + '*')
            if keys:
                status = self._client.delete(*keys)
        else:
            status = self._client.flushdb()
        return status

    def inc(self, key, delta=1):
        return self._client.incr(name=self.key_prefix + key, amount=delta)

    def dec(self, key, delta=1):
        return self._client.decr(name=self.key_prefix + key, amount=delta)


class FileSystemCache(BaseCache):

    """A cache that stores the items on the file system.  This cache depends
    on being the only user of the `cache_dir`.  Make absolutely sure that
    nobody but this cache stores files there or otherwise the cache will
    randomly delete files therein.

    :param cache_dir: the directory where cache files are stored.
    :param threshold: the maximum number of items the cache stores before
                      it starts deleting some.
    :param default_timeout: the default timeout that is used if no timeout is
                            specified on :meth:`~BaseCache.set`. A timeout of
                            0 indicates that the cache never expires.
    :param mode: the file mode wanted for the cache files, default 0600
    """

    #: used for temporary files by the FileSystemCache
    _fs_transaction_suffix = '.__wz_cache'

    def __init__(self, cache_dir, threshold=500, default_timeout=300,
                 mode=0o600):
        BaseCache.__init__(self, default_timeout)
        self._path = cache_dir
        self._threshold = threshold
        self._mode = mode

        try:
            os.makedirs(self._path)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    def _list_dir(self):
        """return a list of (fully qualified) cache filenames
        """
        return [os.path.join(self._path, fn) for fn in os.listdir(self._path)
                if not fn.endswith(self._fs_transaction_suffix)]

    def _prune(self):
        entries = self._list_dir()
        if len(entries) > self._threshold:
            now = time()
            try:
                for idx, fname in enumerate(entries):
                    remove = False
                    with open(fname, 'rb') as f:
                        expires = pickle.load(f)
                    remove = (expires != 0 and expires <= now) or idx % 3 == 0

                    if remove:
                        os.remove(fname)
            except (IOError, OSError):
                pass

    def clear(self):
        for fname in self._list_dir():
            try:
                os.remove(fname)
            except (IOError, OSError):
                return False
        return True

    def _get_filename(self, key):
        if isinstance(key, text_type):
            key = key.encode('utf-8')  # XXX unicode review
        hash = md5(key).hexdigest()
        return os.path.join(self._path, hash)

    def get(self, key):
        filename = self._get_filename(key)
        try:
            with open(filename, 'rb') as f:
                pickle_time = pickle.load(f)
                if pickle_time == 0 or pickle_time >= time():
                    return pickle.load(f)
                else:
                    os.remove(filename)
                    return None
        except (IOError, OSError, pickle.PickleError):
            return None

    def add(self, key, value, timeout=None):
        filename = self._get_filename(key)
        if not os.path.exists(filename):
            return self.set(key, value, timeout)
        return False

    def set(self, key, value, timeout=None):
        if timeout is None:
            timeout = int(time() + self.default_timeout)
        elif timeout != 0:
            timeout = int(time() + timeout)
        filename = self._get_filename(key)
        self._prune()
        try:
            fd, tmp = tempfile.mkstemp(suffix=self._fs_transaction_suffix,
                                       dir=self._path)
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(timeout, f, 1)
                pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
            rename(tmp, filename)
            os.chmod(filename, self._mode)
        except (IOError, OSError):
            return False
        else:
            return True

    def delete(self, key):
        try:
            os.remove(self._get_filename(key))
        except (IOError, OSError):
            return False
        else:
            return True

    def has(self, key):
        filename = self._get_filename(key)
        try:
            with open(filename, 'rb') as f:
                pickle_time = pickle.load(f)
                if pickle_time == 0 or pickle_time >= time():
                    return True
                else:
                    os.remove(filename)
                    return False
        except (IOError, OSError, pickle.PickleError):
            return False
