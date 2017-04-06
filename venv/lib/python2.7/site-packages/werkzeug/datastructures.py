# -*- coding: utf-8 -*-
"""
    werkzeug.datastructures
    ~~~~~~~~~~~~~~~~~~~~~~~

    This module provides mixins and classes with an immutable interface.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import codecs
import mimetypes
from copy import deepcopy
from itertools import repeat

from werkzeug._internal import _missing, _empty_stream
from werkzeug._compat import iterkeys, itervalues, iteritems, iterlists, \
    PY2, text_type, integer_types, string_types, make_literal_wrapper, \
    to_native
from werkzeug.filesystem import get_filesystem_encoding


_locale_delim_re = re.compile(r'[_-]')


def is_immutable(self):
    raise TypeError('%r objects are immutable' % self.__class__.__name__)


def iter_multi_items(mapping):
    """Iterates over the items of a mapping yielding keys and values
    without dropping any from more complex structures.
    """
    if isinstance(mapping, MultiDict):
        for item in iteritems(mapping, multi=True):
            yield item
    elif isinstance(mapping, dict):
        for key, value in iteritems(mapping):
            if isinstance(value, (tuple, list)):
                for value in value:
                    yield key, value
            else:
                yield key, value
    else:
        for item in mapping:
            yield item


def native_itermethods(names):
    if not PY2:
        return lambda x: x

    def setmethod(cls, name):
        itermethod = getattr(cls, name)
        setattr(cls, 'iter%s' % name, itermethod)
        listmethod = lambda self, *a, **kw: list(itermethod(self, *a, **kw))
        listmethod.__doc__ = \
            'Like :py:meth:`iter%s`, but returns a list.' % name
        setattr(cls, name, listmethod)

    def wrap(cls):
        for name in names:
            setmethod(cls, name)
        return cls
    return wrap


class ImmutableListMixin(object):

    """Makes a :class:`list` immutable.

    .. versionadded:: 0.5

    :private:
    """

    _hash_cache = None

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(tuple(self))
        return rv

    def __reduce_ex__(self, protocol):
        return type(self), (list(self),)

    def __delitem__(self, key):
        is_immutable(self)

    def __delslice__(self, i, j):
        is_immutable(self)

    def __iadd__(self, other):
        is_immutable(self)
    __imul__ = __iadd__

    def __setitem__(self, key, value):
        is_immutable(self)

    def __setslice__(self, i, j, value):
        is_immutable(self)

    def append(self, item):
        is_immutable(self)
    remove = append

    def extend(self, iterable):
        is_immutable(self)

    def insert(self, pos, value):
        is_immutable(self)

    def pop(self, index=-1):
        is_immutable(self)

    def reverse(self):
        is_immutable(self)

    def sort(self, cmp=None, key=None, reverse=None):
        is_immutable(self)


class ImmutableList(ImmutableListMixin, list):

    """An immutable :class:`list`.

    .. versionadded:: 0.5

    :private:
    """

    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            list.__repr__(self),
        )


class ImmutableDictMixin(object):

    """Makes a :class:`dict` immutable.

    .. versionadded:: 0.5

    :private:
    """
    _hash_cache = None

    @classmethod
    def fromkeys(cls, keys, value=None):
        instance = super(cls, cls).__new__(cls)
        instance.__init__(zip(keys, repeat(value)))
        return instance

    def __reduce_ex__(self, protocol):
        return type(self), (dict(self),)

    def _iter_hashitems(self):
        return iteritems(self)

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(frozenset(self._iter_hashitems()))
        return rv

    def setdefault(self, key, default=None):
        is_immutable(self)

    def update(self, *args, **kwargs):
        is_immutable(self)

    def pop(self, key, default=None):
        is_immutable(self)

    def popitem(self):
        is_immutable(self)

    def __setitem__(self, key, value):
        is_immutable(self)

    def __delitem__(self, key):
        is_immutable(self)

    def clear(self):
        is_immutable(self)


class ImmutableMultiDictMixin(ImmutableDictMixin):

    """Makes a :class:`MultiDict` immutable.

    .. versionadded:: 0.5

    :private:
    """

    def __reduce_ex__(self, protocol):
        return type(self), (list(iteritems(self, multi=True)),)

    def _iter_hashitems(self):
        return iteritems(self, multi=True)

    def add(self, key, value):
        is_immutable(self)

    def popitemlist(self):
        is_immutable(self)

    def poplist(self, key):
        is_immutable(self)

    def setlist(self, key, new_list):
        is_immutable(self)

    def setlistdefault(self, key, default_list=None):
        is_immutable(self)


class UpdateDictMixin(object):

    """Makes dicts call `self.on_update` on modifications.

    .. versionadded:: 0.5

    :private:
    """

    on_update = None

    def calls_update(name):
        def oncall(self, *args, **kw):
            rv = getattr(super(UpdateDictMixin, self), name)(*args, **kw)
            if self.on_update is not None:
                self.on_update(self)
            return rv
        oncall.__name__ = name
        return oncall

    def setdefault(self, key, default=None):
        modified = key not in self
        rv = super(UpdateDictMixin, self).setdefault(key, default)
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    def pop(self, key, default=_missing):
        modified = key in self
        if default is _missing:
            rv = super(UpdateDictMixin, self).pop(key)
        else:
            rv = super(UpdateDictMixin, self).pop(key, default)
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    __setitem__ = calls_update('__setitem__')
    __delitem__ = calls_update('__delitem__')
    clear = calls_update('clear')
    popitem = calls_update('popitem')
    update = calls_update('update')
    del calls_update


class TypeConversionDict(dict):

    """Works like a regular dict but the :meth:`get` method can perform
    type conversions.  :class:`MultiDict` and :class:`CombinedMultiDict`
    are subclasses of this class and provide the same feature.

    .. versionadded:: 0.5
    """

    def get(self, key, default=None, type=None):
        """Return the default value if the requested data doesn't exist.
        If `type` is provided and is a callable it should convert the value,
        return it or raise a :exc:`ValueError` if that is not possible.  In
        this case the function will return the default as if the value was not
        found:

        >>> d = TypeConversionDict(foo='42', bar='blub')
        >>> d.get('foo', type=int)
        42
        >>> d.get('bar', -1, type=int)
        -1

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key can't
                        be looked up.  If not further specified `None` is
                        returned.
        :param type: A callable that is used to cast the value in the
                     :class:`MultiDict`.  If a :exc:`ValueError` is raised
                     by this callable the default value is returned.
        """
        try:
            rv = self[key]
            if type is not None:
                rv = type(rv)
        except (KeyError, ValueError):
            rv = default
        return rv


class ImmutableTypeConversionDict(ImmutableDictMixin, TypeConversionDict):

    """Works like a :class:`TypeConversionDict` but does not support
    modifications.

    .. versionadded:: 0.5
    """

    def copy(self):
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return TypeConversionDict(self)

    def __copy__(self):
        return self


@native_itermethods(['keys', 'values', 'items', 'lists', 'listvalues'])
class MultiDict(TypeConversionDict):

    """A :class:`MultiDict` is a dictionary subclass customized to deal with
    multiple values for the same key which is for example used by the parsing
    functions in the wrappers.  This is necessary because some HTML form
    elements pass multiple values for the same key.

    :class:`MultiDict` implements all standard dictionary methods.
    Internally, it saves all values for a key as a list, but the standard dict
    access methods will only return the first value for a key. If you want to
    gain access to the other values, too, you have to use the `list` methods as
    explained below.

    Basic Usage:

    >>> d = MultiDict([('a', 'b'), ('a', 'c')])
    >>> d
    MultiDict([('a', 'b'), ('a', 'c')])
    >>> d['a']
    'b'
    >>> d.getlist('a')
    ['b', 'c']
    >>> 'a' in d
    True

    It behaves like a normal dict thus all dict functions will only return the
    first value when multiple values for one key are found.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for HTTP
    exceptions.

    A :class:`MultiDict` can be constructed from an iterable of
    ``(key, value)`` tuples, a dict, a :class:`MultiDict` or from Werkzeug 0.2
    onwards some keyword parameters.

    :param mapping: the initial value for the :class:`MultiDict`.  Either a
                    regular dict, an iterable of ``(key, value)`` tuples
                    or `None`.
    """

    def __init__(self, mapping=None):
        if isinstance(mapping, MultiDict):
            dict.__init__(self, ((k, l[:]) for k, l in iterlists(mapping)))
        elif isinstance(mapping, dict):
            tmp = {}
            for key, value in iteritems(mapping):
                if isinstance(value, (tuple, list)):
                    if len(value) == 0:
                        continue
                    value = list(value)
                else:
                    value = [value]
                tmp[key] = value
            dict.__init__(self, tmp)
        else:
            tmp = {}
            for key, value in mapping or ():
                tmp.setdefault(key, []).append(value)
            dict.__init__(self, tmp)

    def __getstate__(self):
        return dict(self.lists())

    def __setstate__(self, value):
        dict.clear(self)
        dict.update(self, value)

    def __getitem__(self, key):
        """Return the first data value for this key;
        raises KeyError if not found.

        :param key: The key to be looked up.
        :raise KeyError: if the key does not exist.
        """
        if key in self:
            lst = dict.__getitem__(self, key)
            if len(lst) > 0:
                return lst[0]
        raise exceptions.BadRequestKeyError(key)

    def __setitem__(self, key, value):
        """Like :meth:`add` but removes an existing key first.

        :param key: the key for the value.
        :param value: the value to set.
        """
        dict.__setitem__(self, key, [value])

    def add(self, key, value):
        """Adds a new value for the key.

        .. versionadded:: 0.6

        :param key: the key for the value.
        :param value: the value to add.
        """
        dict.setdefault(self, key, []).append(value)

    def getlist(self, key, type=None):
        """Return the list of items for a given key. If that key is not in the
        `MultiDict`, the return value will be an empty list.  Just as `get`
        `getlist` accepts a `type` parameter.  All items will be converted
        with the callable defined there.

        :param key: The key to be looked up.
        :param type: A callable that is used to cast the value in the
                     :class:`MultiDict`.  If a :exc:`ValueError` is raised
                     by this callable the value will be removed from the list.
        :return: a :class:`list` of all the values for the key.
        """
        try:
            rv = dict.__getitem__(self, key)
        except KeyError:
            return []
        if type is None:
            return list(rv)
        result = []
        for item in rv:
            try:
                result.append(type(item))
            except ValueError:
                pass
        return result

    def setlist(self, key, new_list):
        """Remove the old values for a key and add new ones.  Note that the list
        you pass the values in will be shallow-copied before it is inserted in
        the dictionary.

        >>> d = MultiDict()
        >>> d.setlist('foo', ['1', '2'])
        >>> d['foo']
        '1'
        >>> d.getlist('foo')
        ['1', '2']

        :param key: The key for which the values are set.
        :param new_list: An iterable with the new values for the key.  Old values
                         are removed first.
        """
        dict.__setitem__(self, key, list(new_list))

    def setdefault(self, key, default=None):
        """Returns the value for the key if it is in the dict, otherwise it
        returns `default` and sets that value for `key`.

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key is not
                        in the dict.  If not further specified it's `None`.
        """
        if key not in self:
            self[key] = default
        else:
            default = self[key]
        return default

    def setlistdefault(self, key, default_list=None):
        """Like `setdefault` but sets multiple values.  The list returned
        is not a copy, but the list that is actually used internally.  This
        means that you can put new values into the dict by appending items
        to the list:

        >>> d = MultiDict({"foo": 1})
        >>> d.setlistdefault("foo").extend([2, 3])
        >>> d.getlist("foo")
        [1, 2, 3]

        :param key: The key to be looked up.
        :param default: An iterable of default values.  It is either copied
                        (in case it was a list) or converted into a list
                        before returned.
        :return: a :class:`list`
        """
        if key not in self:
            default_list = list(default_list or ())
            dict.__setitem__(self, key, default_list)
        else:
            default_list = dict.__getitem__(self, key)
        return default_list

    def items(self, multi=False):
        """Return an iterator of ``(key, value)`` pairs.

        :param multi: If set to `True` the iterator returned will have a pair
                      for each value of each key.  Otherwise it will only
                      contain pairs for the first value of each key.
        """

        for key, values in iteritems(dict, self):
            if multi:
                for value in values:
                    yield key, value
            else:
                yield key, values[0]

    def lists(self):
        """Return a list of ``(key, values)`` pairs, where values is the list
        of all values associated with the key."""

        for key, values in iteritems(dict, self):
            yield key, list(values)

    def keys(self):
        return iterkeys(dict, self)

    __iter__ = keys

    def values(self):
        """Returns an iterator of the first value on every key's value list."""
        for values in itervalues(dict, self):
            yield values[0]

    def listvalues(self):
        """Return an iterator of all values associated with a key.  Zipping
        :meth:`keys` and this is the same as calling :meth:`lists`:

        >>> d = MultiDict({"foo": [1, 2, 3]})
        >>> zip(d.keys(), d.listvalues()) == d.lists()
        True
        """

        return itervalues(dict, self)

    def copy(self):
        """Return a shallow copy of this object."""
        return self.__class__(self)

    def deepcopy(self, memo=None):
        """Return a deep copy of this object."""
        return self.__class__(deepcopy(self.to_dict(flat=False), memo))

    def to_dict(self, flat=True):
        """Return the contents as regular dict.  If `flat` is `True` the
        returned dict will only have the first item present, if `flat` is
        `False` all values will be returned as lists.

        :param flat: If set to `False` the dict returned will have lists
                     with all the values in it.  Otherwise it will only
                     contain the first value for each key.
        :return: a :class:`dict`
        """
        if flat:
            return dict(iteritems(self))
        return dict(self.lists())

    def update(self, other_dict):
        """update() extends rather than replaces existing key lists:

        >>> a = MultiDict({'x': 1})
        >>> b = MultiDict({'x': 2, 'y': 3})
        >>> a.update(b)
        >>> a
        MultiDict([('y', 3), ('x', 1), ('x', 2)])

        If the value list for a key in ``other_dict`` is empty, no new values
        will be added to the dict and the key will not be created:

        >>> x = {'empty_list': []}
        >>> y = MultiDict()
        >>> y.update(x)
        >>> y
        MultiDict([])
        """
        for key, value in iter_multi_items(other_dict):
            MultiDict.add(self, key, value)

    def pop(self, key, default=_missing):
        """Pop the first item for a list on the dict.  Afterwards the
        key is removed from the dict, so additional values are discarded:

        >>> d = MultiDict({"foo": [1, 2, 3]})
        >>> d.pop("foo")
        1
        >>> "foo" in d
        False

        :param key: the key to pop.
        :param default: if provided the value to return if the key was
                        not in the dictionary.
        """
        try:
            lst = dict.pop(self, key)

            if len(lst) == 0:
                raise exceptions.BadRequestKeyError()

            return lst[0]
        except KeyError as e:
            if default is not _missing:
                return default
            raise exceptions.BadRequestKeyError(str(e))

    def popitem(self):
        """Pop an item from the dict."""
        try:
            item = dict.popitem(self)

            if len(item[1]) == 0:
                raise exceptions.BadRequestKeyError()

            return (item[0], item[1][0])
        except KeyError as e:
            raise exceptions.BadRequestKeyError(str(e))

    def poplist(self, key):
        """Pop the list for a key from the dict.  If the key is not in the dict
        an empty list is returned.

        .. versionchanged:: 0.5
           If the key does no longer exist a list is returned instead of
           raising an error.
        """
        return dict.pop(self, key, [])

    def popitemlist(self):
        """Pop a ``(key, list)`` tuple from the dict."""
        try:
            return dict.popitem(self)
        except KeyError as e:
            raise exceptions.BadRequestKeyError(str(e))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.deepcopy(memo=memo)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, list(iteritems(self, multi=True)))


class _omd_bucket(object):

    """Wraps values in the :class:`OrderedMultiDict`.  This makes it
    possible to keep an order over multiple different keys.  It requires
    a lot of extra memory and slows down access a lot, but makes it
    possible to access elements in O(1) and iterate in O(n).
    """
    __slots__ = ('prev', 'key', 'value', 'next')

    def __init__(self, omd, key, value):
        self.prev = omd._last_bucket
        self.key = key
        self.value = value
        self.next = None

        if omd._first_bucket is None:
            omd._first_bucket = self
        if omd._last_bucket is not None:
            omd._last_bucket.next = self
        omd._last_bucket = self

    def unlink(self, omd):
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        if omd._first_bucket is self:
            omd._first_bucket = self.next
        if omd._last_bucket is self:
            omd._last_bucket = self.prev


@native_itermethods(['keys', 'values', 'items', 'lists', 'listvalues'])
class OrderedMultiDict(MultiDict):

    """Works like a regular :class:`MultiDict` but preserves the
    order of the fields.  To convert the ordered multi dict into a
    list you can use the :meth:`items` method and pass it ``multi=True``.

    In general an :class:`OrderedMultiDict` is an order of magnitude
    slower than a :class:`MultiDict`.

    .. admonition:: note

       Due to a limitation in Python you cannot convert an ordered
       multi dict into a regular dict by using ``dict(multidict)``.
       Instead you have to use the :meth:`to_dict` method, otherwise
       the internal bucket objects are exposed.
    """

    def __init__(self, mapping=None):
        dict.__init__(self)
        self._first_bucket = self._last_bucket = None
        if mapping is not None:
            OrderedMultiDict.update(self, mapping)

    def __eq__(self, other):
        if not isinstance(other, MultiDict):
            return NotImplemented
        if isinstance(other, OrderedMultiDict):
            iter1 = iteritems(self, multi=True)
            iter2 = iteritems(other, multi=True)
            try:
                for k1, v1 in iter1:
                    k2, v2 = next(iter2)
                    if k1 != k2 or v1 != v2:
                        return False
            except StopIteration:
                return False
            try:
                next(iter2)
            except StopIteration:
                return True
            return False
        if len(self) != len(other):
            return False
        for key, values in iterlists(self):
            if other.getlist(key) != values:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce_ex__(self, protocol):
        return type(self), (list(iteritems(self, multi=True)),)

    def __getstate__(self):
        return list(iteritems(self, multi=True))

    def __setstate__(self, values):
        dict.clear(self)
        for key, value in values:
            self.add(key, value)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)[0].value
        raise exceptions.BadRequestKeyError(key)

    def __setitem__(self, key, value):
        self.poplist(key)
        self.add(key, value)

    def __delitem__(self, key):
        self.pop(key)

    def keys(self):
        return (key for key, value in iteritems(self))

    __iter__ = keys

    def values(self):
        return (value for key, value in iteritems(self))

    def items(self, multi=False):
        ptr = self._first_bucket
        if multi:
            while ptr is not None:
                yield ptr.key, ptr.value
                ptr = ptr.next
        else:
            returned_keys = set()
            while ptr is not None:
                if ptr.key not in returned_keys:
                    returned_keys.add(ptr.key)
                    yield ptr.key, ptr.value
                ptr = ptr.next

    def lists(self):
        returned_keys = set()
        ptr = self._first_bucket
        while ptr is not None:
            if ptr.key not in returned_keys:
                yield ptr.key, self.getlist(ptr.key)
                returned_keys.add(ptr.key)
            ptr = ptr.next

    def listvalues(self):
        for key, values in iterlists(self):
            yield values

    def add(self, key, value):
        dict.setdefault(self, key, []).append(_omd_bucket(self, key, value))

    def getlist(self, key, type=None):
        try:
            rv = dict.__getitem__(self, key)
        except KeyError:
            return []
        if type is None:
            return [x.value for x in rv]
        result = []
        for item in rv:
            try:
                result.append(type(item.value))
            except ValueError:
                pass
        return result

    def setlist(self, key, new_list):
        self.poplist(key)
        for value in new_list:
            self.add(key, value)

    def setlistdefault(self, key, default_list=None):
        raise TypeError('setlistdefault is unsupported for '
                        'ordered multi dicts')

    def update(self, mapping):
        for key, value in iter_multi_items(mapping):
            OrderedMultiDict.add(self, key, value)

    def poplist(self, key):
        buckets = dict.pop(self, key, ())
        for bucket in buckets:
            bucket.unlink(self)
        return [x.value for x in buckets]

    def pop(self, key, default=_missing):
        try:
            buckets = dict.pop(self, key)
        except KeyError as e:
            if default is not _missing:
                return default
            raise exceptions.BadRequestKeyError(str(e))
        for bucket in buckets:
            bucket.unlink(self)
        return buckets[0].value

    def popitem(self):
        try:
            key, buckets = dict.popitem(self)
        except KeyError as e:
            raise exceptions.BadRequestKeyError(str(e))
        for bucket in buckets:
            bucket.unlink(self)
        return key, buckets[0].value

    def popitemlist(self):
        try:
            key, buckets = dict.popitem(self)
        except KeyError as e:
            raise exceptions.BadRequestKeyError(str(e))
        for bucket in buckets:
            bucket.unlink(self)
        return key, [x.value for x in buckets]


def _options_header_vkw(value, kw):
    return dump_options_header(value, dict((k.replace('_', '-'), v)
                                           for k, v in kw.items()))


def _unicodify_header_value(value):
    if isinstance(value, bytes):
        value = value.decode('latin-1')
    if not isinstance(value, text_type):
        value = text_type(value)
    return value


@native_itermethods(['keys', 'values', 'items'])
class Headers(object):

    """An object that stores some headers.  It has a dict-like interface
    but is ordered and can store the same keys multiple times.

    This data structure is useful if you want a nicer way to handle WSGI
    headers which are stored as tuples in a list.

    From Werkzeug 0.3 onwards, the :exc:`KeyError` raised by this class is
    also a subclass of the :class:`~exceptions.BadRequest` HTTP exception
    and will render a page for a ``400 BAD REQUEST`` if caught in a
    catch-all for HTTP exceptions.

    Headers is mostly compatible with the Python :class:`wsgiref.headers.Headers`
    class, with the exception of `__getitem__`.  :mod:`wsgiref` will return
    `None` for ``headers['missing']``, whereas :class:`Headers` will raise
    a :class:`KeyError`.

    To create a new :class:`Headers` object pass it a list or dict of headers
    which are used as default values.  This does not reuse the list passed
    to the constructor for internal usage.

    :param defaults: The list of default values for the :class:`Headers`.

    .. versionchanged:: 0.9
       This data structure now stores unicode values similar to how the
       multi dicts do it.  The main difference is that bytes can be set as
       well which will automatically be latin1 decoded.

    .. versionchanged:: 0.9
       The :meth:`linked` function was removed without replacement as it
       was an API that does not support the changes to the encoding model.
    """

    def __init__(self, defaults=None):
        self._list = []
        if defaults is not None:
            if isinstance(defaults, (list, Headers)):
                self._list.extend(defaults)
            else:
                self.extend(defaults)

    def __getitem__(self, key, _get_mode=False):
        if not _get_mode:
            if isinstance(key, integer_types):
                return self._list[key]
            elif isinstance(key, slice):
                return self.__class__(self._list[key])
        if not isinstance(key, string_types):
            raise exceptions.BadRequestKeyError(key)
        ikey = key.lower()
        for k, v in self._list:
            if k.lower() == ikey:
                return v
        # micro optimization: if we are in get mode we will catch that
        # exception one stack level down so we can raise a standard
        # key error instead of our special one.
        if _get_mode:
            raise KeyError()
        raise exceptions.BadRequestKeyError(key)

    def __eq__(self, other):
        return other.__class__ is self.__class__ and \
            set(other._list) == set(self._list)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get(self, key, default=None, type=None, as_bytes=False):
        """Return the default value if the requested data doesn't exist.
        If `type` is provided and is a callable it should convert the value,
        return it or raise a :exc:`ValueError` if that is not possible.  In
        this case the function will return the default as if the value was not
        found:

        >>> d = Headers([('Content-Length', '42')])
        >>> d.get('Content-Length', type=int)
        42

        If a headers object is bound you must not add unicode strings
        because no encoding takes place.

        .. versionadded:: 0.9
           Added support for `as_bytes`.

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key can't
                        be looked up.  If not further specified `None` is
                        returned.
        :param type: A callable that is used to cast the value in the
                     :class:`Headers`.  If a :exc:`ValueError` is raised
                     by this callable the default value is returned.
        :param as_bytes: return bytes instead of unicode strings.
        """
        try:
            rv = self.__getitem__(key, _get_mode=True)
        except KeyError:
            return default
        if as_bytes:
            rv = rv.encode('latin1')
        if type is None:
            return rv
        try:
            return type(rv)
        except ValueError:
            return default

    def getlist(self, key, type=None, as_bytes=False):
        """Return the list of items for a given key. If that key is not in the
        :class:`Headers`, the return value will be an empty list.  Just as
        :meth:`get` :meth:`getlist` accepts a `type` parameter.  All items will
        be converted with the callable defined there.

        .. versionadded:: 0.9
           Added support for `as_bytes`.

        :param key: The key to be looked up.
        :param type: A callable that is used to cast the value in the
                     :class:`Headers`.  If a :exc:`ValueError` is raised
                     by this callable the value will be removed from the list.
        :return: a :class:`list` of all the values for the key.
        :param as_bytes: return bytes instead of unicode strings.
        """
        ikey = key.lower()
        result = []
        for k, v in self:
            if k.lower() == ikey:
                if as_bytes:
                    v = v.encode('latin1')
                if type is not None:
                    try:
                        v = type(v)
                    except ValueError:
                        continue
                result.append(v)
        return result

    def get_all(self, name):
        """Return a list of all the values for the named field.

        This method is compatible with the :mod:`wsgiref`
        :meth:`~wsgiref.headers.Headers.get_all` method.
        """
        return self.getlist(name)

    def items(self, lower=False):
        for key, value in self:
            if lower:
                key = key.lower()
            yield key, value

    def keys(self, lower=False):
        for key, _ in iteritems(self, lower):
            yield key

    def values(self):
        for _, value in iteritems(self):
            yield value

    def extend(self, iterable):
        """Extend the headers with a dict or an iterable yielding keys and
        values.
        """
        if isinstance(iterable, dict):
            for key, value in iteritems(iterable):
                if isinstance(value, (tuple, list)):
                    for v in value:
                        self.add(key, v)
                else:
                    self.add(key, value)
        else:
            for key, value in iterable:
                self.add(key, value)

    def __delitem__(self, key, _index_operation=True):
        if _index_operation and isinstance(key, (integer_types, slice)):
            del self._list[key]
            return
        key = key.lower()
        new = []
        for k, v in self._list:
            if k.lower() != key:
                new.append((k, v))
        self._list[:] = new

    def remove(self, key):
        """Remove a key.

        :param key: The key to be removed.
        """
        return self.__delitem__(key, _index_operation=False)

    def pop(self, key=None, default=_missing):
        """Removes and returns a key or index.

        :param key: The key to be popped.  If this is an integer the item at
                    that position is removed, if it's a string the value for
                    that key is.  If the key is omitted or `None` the last
                    item is removed.
        :return: an item.
        """
        if key is None:
            return self._list.pop()
        if isinstance(key, integer_types):
            return self._list.pop(key)
        try:
            rv = self[key]
            self.remove(key)
        except KeyError:
            if default is not _missing:
                return default
            raise
        return rv

    def popitem(self):
        """Removes a key or index and returns a (key, value) item."""
        return self.pop()

    def __contains__(self, key):
        """Check if a key is present."""
        try:
            self.__getitem__(key, _get_mode=True)
        except KeyError:
            return False
        return True

    has_key = __contains__

    def __iter__(self):
        """Yield ``(key, value)`` tuples."""
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def add(self, _key, _value, **kw):
        """Add a new header tuple to the list.

        Keyword arguments can specify additional parameters for the header
        value, with underscores converted to dashes::

        >>> d = Headers()
        >>> d.add('Content-Type', 'text/plain')
        >>> d.add('Content-Disposition', 'attachment', filename='foo.png')

        The keyword argument dumping uses :func:`dump_options_header`
        behind the scenes.

        .. versionadded:: 0.4.1
            keyword arguments were added for :mod:`wsgiref` compatibility.
        """
        if kw:
            _value = _options_header_vkw(_value, kw)
        _value = _unicodify_header_value(_value)
        self._validate_value(_value)
        self._list.append((_key, _value))

    def _validate_value(self, value):
        if not isinstance(value, text_type):
            raise TypeError('Value should be unicode.')
        if u'\n' in value or u'\r' in value:
            raise ValueError('Detected newline in header value.  This is '
                             'a potential security problem')

    def add_header(self, _key, _value, **_kw):
        """Add a new header tuple to the list.

        An alias for :meth:`add` for compatibility with the :mod:`wsgiref`
        :meth:`~wsgiref.headers.Headers.add_header` method.
        """
        self.add(_key, _value, **_kw)

    def clear(self):
        """Clears all headers."""
        del self._list[:]

    def set(self, _key, _value, **kw):
        """Remove all header tuples for `key` and add a new one.  The newly
        added key either appears at the end of the list if there was no
        entry or replaces the first one.

        Keyword arguments can specify additional parameters for the header
        value, with underscores converted to dashes.  See :meth:`add` for
        more information.

        .. versionchanged:: 0.6.1
           :meth:`set` now accepts the same arguments as :meth:`add`.

        :param key: The key to be inserted.
        :param value: The value to be inserted.
        """
        if kw:
            _value = _options_header_vkw(_value, kw)
        _value = _unicodify_header_value(_value)
        self._validate_value(_value)
        if not self._list:
            self._list.append((_key, _value))
            return
        listiter = iter(self._list)
        ikey = _key.lower()
        for idx, (old_key, old_value) in enumerate(listiter):
            if old_key.lower() == ikey:
                # replace first ocurrence
                self._list[idx] = (_key, _value)
                break
        else:
            self._list.append((_key, _value))
            return
        self._list[idx + 1:] = [t for t in listiter if t[0].lower() != ikey]

    def setdefault(self, key, value):
        """Returns the value for the key if it is in the dict, otherwise it
        returns `default` and sets that value for `key`.

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key is not
                        in the dict.  If not further specified it's `None`.
        """
        if key in self:
            return self[key]
        self.set(key, value)
        return value

    def __setitem__(self, key, value):
        """Like :meth:`set` but also supports index/slice based setting."""
        if isinstance(key, (slice, integer_types)):
            if isinstance(key, integer_types):
                value = [value]
            value = [(k, _unicodify_header_value(v)) for (k, v) in value]
            [self._validate_value(v) for (k, v) in value]
            if isinstance(key, integer_types):
                self._list[key] = value[0]
            else:
                self._list[key] = value
        else:
            self.set(key, value)

    def to_list(self, charset='iso-8859-1'):
        """Convert the headers into a list suitable for WSGI."""
        from warnings import warn
        warn(DeprecationWarning('Method removed, use to_wsgi_list instead'),
             stacklevel=2)
        return self.to_wsgi_list()

    def to_wsgi_list(self):
        """Convert the headers into a list suitable for WSGI.

        The values are byte strings in Python 2 converted to latin1 and unicode
        strings in Python 3 for the WSGI server to encode.

        :return: list
        """
        if PY2:
            return [(to_native(k), v.encode('latin1')) for k, v in self]
        return list(self)

    def copy(self):
        return self.__class__(self._list)

    def __copy__(self):
        return self.copy()

    def __str__(self):
        """Returns formatted headers suitable for HTTP transmission."""
        strs = []
        for key, value in self.to_wsgi_list():
            strs.append('%s: %s' % (key, value))
        strs.append('\r\n')
        return '\r\n'.join(strs)

    def __repr__(self):
        return '%s(%r)' % (
            self.__class__.__name__,
            list(self)
        )


class ImmutableHeadersMixin(object):

    """Makes a :class:`Headers` immutable.  We do not mark them as
    hashable though since the only usecase for this datastructure
    in Werkzeug is a view on a mutable structure.

    .. versionadded:: 0.5

    :private:
    """

    def __delitem__(self, key):
        is_immutable(self)

    def __setitem__(self, key, value):
        is_immutable(self)
    set = __setitem__

    def add(self, item):
        is_immutable(self)
    remove = add_header = add

    def extend(self, iterable):
        is_immutable(self)

    def insert(self, pos, value):
        is_immutable(self)

    def pop(self, index=-1):
        is_immutable(self)

    def popitem(self):
        is_immutable(self)

    def setdefault(self, key, default):
        is_immutable(self)


class EnvironHeaders(ImmutableHeadersMixin, Headers):

    """Read only version of the headers from a WSGI environment.  This
    provides the same interface as `Headers` and is constructed from
    a WSGI environment.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for
    HTTP exceptions.
    """

    def __init__(self, environ):
        self.environ = environ

    def __eq__(self, other):
        return self.environ is other.environ

    def __getitem__(self, key, _get_mode=False):
        # _get_mode is a no-op for this class as there is no index but
        # used because get() calls it.
        key = key.upper().replace('-', '_')
        if key in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            return _unicodify_header_value(self.environ[key])
        return _unicodify_header_value(self.environ['HTTP_' + key])

    def __len__(self):
        # the iter is necessary because otherwise list calls our
        # len which would call list again and so forth.
        return len(list(iter(self)))

    def __iter__(self):
        for key, value in iteritems(self.environ):
            if key.startswith('HTTP_') and key not in \
               ('HTTP_CONTENT_TYPE', 'HTTP_CONTENT_LENGTH'):
                yield (key[5:].replace('_', '-').title(),
                       _unicodify_header_value(value))
            elif key in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                yield (key.replace('_', '-').title(),
                       _unicodify_header_value(value))

    def copy(self):
        raise TypeError('cannot create %r copies' % self.__class__.__name__)


@native_itermethods(['keys', 'values', 'items', 'lists', 'listvalues'])
class CombinedMultiDict(ImmutableMultiDictMixin, MultiDict):

    """A read only :class:`MultiDict` that you can pass multiple :class:`MultiDict`
    instances as sequence and it will combine the return values of all wrapped
    dicts:

    >>> from werkzeug.datastructures import CombinedMultiDict, MultiDict
    >>> post = MultiDict([('foo', 'bar')])
    >>> get = MultiDict([('blub', 'blah')])
    >>> combined = CombinedMultiDict([get, post])
    >>> combined['foo']
    'bar'
    >>> combined['blub']
    'blah'

    This works for all read operations and will raise a `TypeError` for
    methods that usually change data which isn't possible.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for HTTP
    exceptions.
    """

    def __reduce_ex__(self, protocol):
        return type(self), (self.dicts,)

    def __init__(self, dicts=None):
        self.dicts = dicts or []

    @classmethod
    def fromkeys(cls):
        raise TypeError('cannot create %r instances by fromkeys' %
                        cls.__name__)

    def __getitem__(self, key):
        for d in self.dicts:
            if key in d:
                return d[key]
        raise exceptions.BadRequestKeyError(key)

    def get(self, key, default=None, type=None):
        for d in self.dicts:
            if key in d:
                if type is not None:
                    try:
                        return type(d[key])
                    except ValueError:
                        continue
                return d[key]
        return default

    def getlist(self, key, type=None):
        rv = []
        for d in self.dicts:
            rv.extend(d.getlist(key, type))
        return rv

    def _keys_impl(self):
        """This function exists so __len__ can be implemented more efficiently,
        saving one list creation from an iterator.

        Using this for Python 2's ``dict.keys`` behavior would be useless since
        `dict.keys` in Python 2 returns a list, while we have a set here.
        """
        rv = set()
        for d in self.dicts:
            rv.update(iterkeys(d))
        return rv

    def keys(self):
        return iter(self._keys_impl())

    __iter__ = keys

    def items(self, multi=False):
        found = set()
        for d in self.dicts:
            for key, value in iteritems(d, multi):
                if multi:
                    yield key, value
                elif key not in found:
                    found.add(key)
                    yield key, value

    def values(self):
        for key, value in iteritems(self):
            yield value

    def lists(self):
        rv = {}
        for d in self.dicts:
            for key, values in iterlists(d):
                rv.setdefault(key, []).extend(values)
        return iteritems(rv)

    def listvalues(self):
        return (x[1] for x in self.lists())

    def copy(self):
        """Return a shallow copy of this object."""
        return self.__class__(self.dicts[:])

    def to_dict(self, flat=True):
        """Return the contents as regular dict.  If `flat` is `True` the
        returned dict will only have the first item present, if `flat` is
        `False` all values will be returned as lists.

        :param flat: If set to `False` the dict returned will have lists
                     with all the values in it.  Otherwise it will only
                     contain the first item for each key.
        :return: a :class:`dict`
        """
        rv = {}
        for d in reversed(self.dicts):
            rv.update(d.to_dict(flat))
        return rv

    def __len__(self):
        return len(self._keys_impl())

    def __contains__(self, key):
        for d in self.dicts:
            if key in d:
                return True
        return False

    has_key = __contains__

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.dicts)


class FileMultiDict(MultiDict):

    """A special :class:`MultiDict` that has convenience methods to add
    files to it.  This is used for :class:`EnvironBuilder` and generally
    useful for unittesting.

    .. versionadded:: 0.5
    """

    def add_file(self, name, file, filename=None, content_type=None):
        """Adds a new file to the dict.  `file` can be a file name or
        a :class:`file`-like or a :class:`FileStorage` object.

        :param name: the name of the field.
        :param file: a filename or :class:`file`-like object
        :param filename: an optional filename
        :param content_type: an optional content type
        """
        if isinstance(file, FileStorage):
            value = file
        else:
            if isinstance(file, string_types):
                if filename is None:
                    filename = file
                file = open(file, 'rb')
            if filename and content_type is None:
                content_type = mimetypes.guess_type(filename)[0] or \
                    'application/octet-stream'
            value = FileStorage(file, filename, name, content_type)

        self.add(name, value)


class ImmutableDict(ImmutableDictMixin, dict):

    """An immutable :class:`dict`.

    .. versionadded:: 0.5
    """

    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            dict.__repr__(self),
        )

    def copy(self):
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return dict(self)

    def __copy__(self):
        return self


class ImmutableMultiDict(ImmutableMultiDictMixin, MultiDict):

    """An immutable :class:`MultiDict`.

    .. versionadded:: 0.5
    """

    def copy(self):
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return MultiDict(self)

    def __copy__(self):
        return self


class ImmutableOrderedMultiDict(ImmutableMultiDictMixin, OrderedMultiDict):

    """An immutable :class:`OrderedMultiDict`.

    .. versionadded:: 0.6
    """

    def _iter_hashitems(self):
        return enumerate(iteritems(self, multi=True))

    def copy(self):
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return OrderedMultiDict(self)

    def __copy__(self):
        return self


@native_itermethods(['values'])
class Accept(ImmutableList):

    """An :class:`Accept` object is just a list subclass for lists of
    ``(value, quality)`` tuples.  It is automatically sorted by quality.

    All :class:`Accept` objects work similar to a list but provide extra
    functionality for working with the data.  Containment checks are
    normalized to the rules of that header:

    >>> a = CharsetAccept([('ISO-8859-1', 1), ('utf-8', 0.7)])
    >>> a.best
    'ISO-8859-1'
    >>> 'iso-8859-1' in a
    True
    >>> 'UTF8' in a
    True
    >>> 'utf7' in a
    False

    To get the quality for an item you can use normal item lookup:

    >>> print a['utf-8']
    0.7
    >>> a['utf7']
    0

    .. versionchanged:: 0.5
       :class:`Accept` objects are forced immutable now.
    """

    def __init__(self, values=()):
        if values is None:
            list.__init__(self)
            self.provided = False
        elif isinstance(values, Accept):
            self.provided = values.provided
            list.__init__(self, values)
        else:
            self.provided = True
            values = [(a, b) for b, a in values]
            values.sort()
            values.reverse()
            list.__init__(self, [(a, b) for b, a in values])

    def _value_matches(self, value, item):
        """Check if a value matches a given accept item."""
        return item == '*' or item.lower() == value.lower()

    def __getitem__(self, key):
        """Besides index lookup (getting item n) you can also pass it a string
        to get the quality for the item.  If the item is not in the list, the
        returned quality is ``0``.
        """
        if isinstance(key, string_types):
            return self.quality(key)
        return list.__getitem__(self, key)

    def quality(self, key):
        """Returns the quality of the key.

        .. versionadded:: 0.6
           In previous versions you had to use the item-lookup syntax
           (eg: ``obj[key]`` instead of ``obj.quality(key)``)
        """
        for item, quality in self:
            if self._value_matches(key, item):
                return quality
        return 0

    def __contains__(self, value):
        for item, quality in self:
            if self._value_matches(value, item):
                return True
        return False

    def __repr__(self):
        return '%s([%s])' % (
            self.__class__.__name__,
            ', '.join('(%r, %s)' % (x, y) for x, y in self)
        )

    def index(self, key):
        """Get the position of an entry or raise :exc:`ValueError`.

        :param key: The key to be looked up.

        .. versionchanged:: 0.5
           This used to raise :exc:`IndexError`, which was inconsistent
           with the list API.
        """
        if isinstance(key, string_types):
            for idx, (item, quality) in enumerate(self):
                if self._value_matches(key, item):
                    return idx
            raise ValueError(key)
        return list.index(self, key)

    def find(self, key):
        """Get the position of an entry or return -1.

        :param key: The key to be looked up.
        """
        try:
            return self.index(key)
        except ValueError:
            return -1

    def values(self):
        """Iterate over all values."""
        for item in self:
            yield item[0]

    def to_header(self):
        """Convert the header set into an HTTP header string."""
        result = []
        for value, quality in self:
            if quality != 1:
                value = '%s;q=%s' % (value, quality)
            result.append(value)
        return ','.join(result)

    def __str__(self):
        return self.to_header()

    def best_match(self, matches, default=None):
        """Returns the best match from a list of possible matches based
        on the quality of the client.  If two items have the same quality,
        the one is returned that comes first.

        :param matches: a list of matches to check for
        :param default: the value that is returned if none match
        """
        best_quality = -1
        result = default
        for server_item in matches:
            for client_item, quality in self:
                if quality <= best_quality:
                    break
                if self._value_matches(server_item, client_item) \
                   and quality > 0:
                    best_quality = quality
                    result = server_item
        return result

    @property
    def best(self):
        """The best match as value."""
        if self:
            return self[0][0]


class MIMEAccept(Accept):

    """Like :class:`Accept` but with special methods and behavior for
    mimetypes.
    """

    def _value_matches(self, value, item):
        def _normalize(x):
            x = x.lower()
            return x == '*' and ('*', '*') or x.split('/', 1)

        # this is from the application which is trusted.  to avoid developer
        # frustration we actually check these for valid values
        if '/' not in value:
            raise ValueError('invalid mimetype %r' % value)
        value_type, value_subtype = _normalize(value)
        if value_type == '*' and value_subtype != '*':
            raise ValueError('invalid mimetype %r' % value)

        if '/' not in item:
            return False
        item_type, item_subtype = _normalize(item)
        if item_type == '*' and item_subtype != '*':
            return False
        return (
            (item_type == item_subtype == '*' or
             value_type == value_subtype == '*') or
            (item_type == value_type and (item_subtype == '*' or
                                          value_subtype == '*' or
                                          item_subtype == value_subtype))
        )

    @property
    def accept_html(self):
        """True if this object accepts HTML."""
        return (
            'text/html' in self or
            'application/xhtml+xml' in self or
            self.accept_xhtml
        )

    @property
    def accept_xhtml(self):
        """True if this object accepts XHTML."""
        return (
            'application/xhtml+xml' in self or
            'application/xml' in self
        )

    @property
    def accept_json(self):
        """True if this object accepts JSON."""
        return 'application/json' in self


class LanguageAccept(Accept):

    """Like :class:`Accept` but with normalization for languages."""

    def _value_matches(self, value, item):
        def _normalize(language):
            return _locale_delim_re.split(language.lower())
        return item == '*' or _normalize(value) == _normalize(item)


class CharsetAccept(Accept):

    """Like :class:`Accept` but with normalization for charsets."""

    def _value_matches(self, value, item):
        def _normalize(name):
            try:
                return codecs.lookup(name).name
            except LookupError:
                return name.lower()
        return item == '*' or _normalize(value) == _normalize(item)


def cache_property(key, empty, type):
    """Return a new property object for a cache header.  Useful if you
    want to add support for a cache extension in a subclass."""
    return property(lambda x: x._get_cache_value(key, empty, type),
                    lambda x, v: x._set_cache_value(key, v, type),
                    lambda x: x._del_cache_value(key),
                    'accessor for %r' % key)


class _CacheControl(UpdateDictMixin, dict):

    """Subclass of a dict that stores values for a Cache-Control header.  It
    has accessors for all the cache-control directives specified in RFC 2616.
    The class does not differentiate between request and response directives.

    Because the cache-control directives in the HTTP header use dashes the
    python descriptors use underscores for that.

    To get a header of the :class:`CacheControl` object again you can convert
    the object into a string or call the :meth:`to_header` method.  If you plan
    to subclass it and add your own items have a look at the sourcecode for
    that class.

    .. versionchanged:: 0.4

       Setting `no_cache` or `private` to boolean `True` will set the implicit
       none-value which is ``*``:

       >>> cc = ResponseCacheControl()
       >>> cc.no_cache = True
       >>> cc
       <ResponseCacheControl 'no-cache'>
       >>> cc.no_cache
       '*'
       >>> cc.no_cache = None
       >>> cc
       <ResponseCacheControl ''>

       In versions before 0.5 the behavior documented here affected the now
       no longer existing `CacheControl` class.
    """

    no_cache = cache_property('no-cache', '*', None)
    no_store = cache_property('no-store', None, bool)
    max_age = cache_property('max-age', -1, int)
    no_transform = cache_property('no-transform', None, None)

    def __init__(self, values=(), on_update=None):
        dict.__init__(self, values or ())
        self.on_update = on_update
        self.provided = values is not None

    def _get_cache_value(self, key, empty, type):
        """Used internally by the accessor properties."""
        if type is bool:
            return key in self
        if key in self:
            value = self[key]
            if value is None:
                return empty
            elif type is not None:
                try:
                    value = type(value)
                except ValueError:
                    pass
            return value

    def _set_cache_value(self, key, value, type):
        """Used internally by the accessor properties."""
        if type is bool:
            if value:
                self[key] = None
            else:
                self.pop(key, None)
        else:
            if value is None:
                self.pop(key)
            elif value is True:
                self[key] = None
            else:
                self[key] = value

    def _del_cache_value(self, key):
        """Used internally by the accessor properties."""
        if key in self:
            del self[key]

    def to_header(self):
        """Convert the stored values into a cache control header."""
        return dump_header(self)

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return '<%s %s>' % (
            self.__class__.__name__,
            " ".join(
                "%s=%r" % (k, v) for k, v in sorted(self.items())
            ),
        )


class RequestCacheControl(ImmutableDictMixin, _CacheControl):

    """A cache control for requests.  This is immutable and gives access
    to all the request-relevant cache control headers.

    To get a header of the :class:`RequestCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionadded:: 0.5
       In previous versions a `CacheControl` class existed that was used
       both for request and response.
    """

    max_stale = cache_property('max-stale', '*', int)
    min_fresh = cache_property('min-fresh', '*', int)
    no_transform = cache_property('no-transform', None, None)
    only_if_cached = cache_property('only-if-cached', None, bool)


class ResponseCacheControl(_CacheControl):

    """A cache control for responses.  Unlike :class:`RequestCacheControl`
    this is mutable and gives access to response-relevant cache control
    headers.

    To get a header of the :class:`ResponseCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionadded:: 0.5
       In previous versions a `CacheControl` class existed that was used
       both for request and response.
    """

    public = cache_property('public', None, bool)
    private = cache_property('private', '*', None)
    must_revalidate = cache_property('must-revalidate', None, bool)
    proxy_revalidate = cache_property('proxy-revalidate', None, bool)
    s_maxage = cache_property('s-maxage', None, None)


# attach cache_property to the _CacheControl as staticmethod
# so that others can reuse it.
_CacheControl.cache_property = staticmethod(cache_property)


class CallbackDict(UpdateDictMixin, dict):

    """A dict that calls a function passed every time something is changed.
    The function is passed the dict instance.
    """

    def __init__(self, initial=None, on_update=None):
        dict.__init__(self, initial or ())
        self.on_update = on_update

    def __repr__(self):
        return '<%s %s>' % (
            self.__class__.__name__,
            dict.__repr__(self)
        )


class HeaderSet(object):

    """Similar to the :class:`ETags` class this implements a set-like structure.
    Unlike :class:`ETags` this is case insensitive and used for vary, allow, and
    content-language headers.

    If not constructed using the :func:`parse_set_header` function the
    instantiation works like this:

    >>> hs = HeaderSet(['foo', 'bar', 'baz'])
    >>> hs
    HeaderSet(['foo', 'bar', 'baz'])
    """

    def __init__(self, headers=None, on_update=None):
        self._headers = list(headers or ())
        self._set = set([x.lower() for x in self._headers])
        self.on_update = on_update

    def add(self, header):
        """Add a new header to the set."""
        self.update((header,))

    def remove(self, header):
        """Remove a header from the set.  This raises an :exc:`KeyError` if the
        header is not in the set.

        .. versionchanged:: 0.5
            In older versions a :exc:`IndexError` was raised instead of a
            :exc:`KeyError` if the object was missing.

        :param header: the header to be removed.
        """
        key = header.lower()
        if key not in self._set:
            raise KeyError(header)
        self._set.remove(key)
        for idx, key in enumerate(self._headers):
            if key.lower() == header:
                del self._headers[idx]
                break
        if self.on_update is not None:
            self.on_update(self)

    def update(self, iterable):
        """Add all the headers from the iterable to the set.

        :param iterable: updates the set with the items from the iterable.
        """
        inserted_any = False
        for header in iterable:
            key = header.lower()
            if key not in self._set:
                self._headers.append(header)
                self._set.add(key)
                inserted_any = True
        if inserted_any and self.on_update is not None:
            self.on_update(self)

    def discard(self, header):
        """Like :meth:`remove` but ignores errors.

        :param header: the header to be discarded.
        """
        try:
            return self.remove(header)
        except KeyError:
            pass

    def find(self, header):
        """Return the index of the header in the set or return -1 if not found.

        :param header: the header to be looked up.
        """
        header = header.lower()
        for idx, item in enumerate(self._headers):
            if item.lower() == header:
                return idx
        return -1

    def index(self, header):
        """Return the index of the header in the set or raise an
        :exc:`IndexError`.

        :param header: the header to be looked up.
        """
        rv = self.find(header)
        if rv < 0:
            raise IndexError(header)
        return rv

    def clear(self):
        """Clear the set."""
        self._set.clear()
        del self._headers[:]
        if self.on_update is not None:
            self.on_update(self)

    def as_set(self, preserve_casing=False):
        """Return the set as real python set type.  When calling this, all
        the items are converted to lowercase and the ordering is lost.

        :param preserve_casing: if set to `True` the items in the set returned
                                will have the original case like in the
                                :class:`HeaderSet`, otherwise they will
                                be lowercase.
        """
        if preserve_casing:
            return set(self._headers)
        return set(self._set)

    def to_header(self):
        """Convert the header set into an HTTP header string."""
        return ', '.join(map(quote_header_value, self._headers))

    def __getitem__(self, idx):
        return self._headers[idx]

    def __delitem__(self, idx):
        rv = self._headers.pop(idx)
        self._set.remove(rv.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __setitem__(self, idx, value):
        old = self._headers[idx]
        self._set.remove(old.lower())
        self._headers[idx] = value
        self._set.add(value.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __contains__(self, header):
        return header.lower() in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._headers)

    def __nonzero__(self):
        return bool(self._set)

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return '%s(%r)' % (
            self.__class__.__name__,
            self._headers
        )


class ETags(object):

    """A set that can be used to check if one etag is present in a collection
    of etags.
    """

    def __init__(self, strong_etags=None, weak_etags=None, star_tag=False):
        self._strong = frozenset(not star_tag and strong_etags or ())
        self._weak = frozenset(weak_etags or ())
        self.star_tag = star_tag

    def as_set(self, include_weak=False):
        """Convert the `ETags` object into a python set.  Per default all the
        weak etags are not part of this set."""
        rv = set(self._strong)
        if include_weak:
            rv.update(self._weak)
        return rv

    def is_weak(self, etag):
        """Check if an etag is weak."""
        return etag in self._weak

    def contains_weak(self, etag):
        """Check if an etag is part of the set including weak and strong tags."""
        return self.is_weak(etag) or self.contains(etag)

    def contains(self, etag):
        """Check if an etag is part of the set ignoring weak tags.
        It is also possible to use the ``in`` operator.

        """
        if self.star_tag:
            return True
        return etag in self._strong

    def contains_raw(self, etag):
        """When passed a quoted tag it will check if this tag is part of the
        set.  If the tag is weak it is checked against weak and strong tags,
        otherwise strong only."""
        etag, weak = unquote_etag(etag)
        if weak:
            return self.contains_weak(etag)
        return self.contains(etag)

    def to_header(self):
        """Convert the etags set into a HTTP header string."""
        if self.star_tag:
            return '*'
        return ', '.join(
            ['"%s"' % x for x in self._strong] +
            ['W/"%s"' % x for x in self._weak]
        )

    def __call__(self, etag=None, data=None, include_weak=False):
        if [etag, data].count(None) != 1:
            raise TypeError('either tag or data required, but at least one')
        if etag is None:
            etag = generate_etag(data)
        if include_weak:
            if etag in self._weak:
                return True
        return etag in self._strong

    def __bool__(self):
        return bool(self.star_tag or self._strong or self._weak)

    __nonzero__ = __bool__

    def __str__(self):
        return self.to_header()

    def __iter__(self):
        return iter(self._strong)

    def __contains__(self, etag):
        return self.contains(etag)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, str(self))


class IfRange(object):

    """Very simple object that represents the `If-Range` header in parsed
    form.  It will either have neither a etag or date or one of either but
    never both.

    .. versionadded:: 0.7
    """

    def __init__(self, etag=None, date=None):
        #: The etag parsed and unquoted.  Ranges always operate on strong
        #: etags so the weakness information is not necessary.
        self.etag = etag
        #: The date in parsed format or `None`.
        self.date = date

    def to_header(self):
        """Converts the object back into an HTTP header."""
        if self.date is not None:
            return http_date(self.date)
        if self.etag is not None:
            return quote_etag(self.etag)
        return ''

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, str(self))


class Range(object):

    """Represents a range header.  All the methods are only supporting bytes
    as unit.  It does store multiple ranges but :meth:`range_for_length` will
    only work if only one range is provided.

    .. versionadded:: 0.7
    """

    def __init__(self, units, ranges):
        #: The units of this range.  Usually "bytes".
        self.units = units
        #: A list of ``(begin, end)`` tuples for the range header provided.
        #: The ranges are non-inclusive.
        self.ranges = ranges

    def range_for_length(self, length):
        """If the range is for bytes, the length is not None and there is
        exactly one range and it is satisfiable it returns a ``(start, stop)``
        tuple, otherwise `None`.
        """
        if self.units != 'bytes' or length is None or len(self.ranges) != 1:
            return None
        start, end = self.ranges[0]
        if end is None:
            end = length
            if start < 0:
                start += length
        if is_byte_range_valid(start, end, length):
            return start, min(end, length)

    def make_content_range(self, length):
        """Creates a :class:`~werkzeug.datastructures.ContentRange` object
        from the current range and given content length.
        """
        rng = self.range_for_length(length)
        if rng is not None:
            return ContentRange(self.units, rng[0], rng[1], length)

    def to_header(self):
        """Converts the object back into an HTTP header."""
        ranges = []
        for begin, end in self.ranges:
            if end is None:
                ranges.append(begin >= 0 and '%s-' % begin or str(begin))
            else:
                ranges.append('%s-%s' % (begin, end - 1))
        return '%s=%s' % (self.units, ','.join(ranges))

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, str(self))


class ContentRange(object):

    """Represents the content range header.

    .. versionadded:: 0.7
    """

    def __init__(self, units, start, stop, length=None, on_update=None):
        assert is_byte_range_valid(start, stop, length), \
            'Bad range provided'
        self.on_update = on_update
        self.set(start, stop, length, units)

    def _callback_property(name):
        def fget(self):
            return getattr(self, name)

        def fset(self, value):
            setattr(self, name, value)
            if self.on_update is not None:
                self.on_update(self)
        return property(fget, fset)

    #: The units to use, usually "bytes"
    units = _callback_property('_units')
    #: The start point of the range or `None`.
    start = _callback_property('_start')
    #: The stop point of the range (non-inclusive) or `None`.  Can only be
    #: `None` if also start is `None`.
    stop = _callback_property('_stop')
    #: The length of the range or `None`.
    length = _callback_property('_length')

    def set(self, start, stop, length=None, units='bytes'):
        """Simple method to update the ranges."""
        assert is_byte_range_valid(start, stop, length), \
            'Bad range provided'
        self._units = units
        self._start = start
        self._stop = stop
        self._length = length
        if self.on_update is not None:
            self.on_update(self)

    def unset(self):
        """Sets the units to `None` which indicates that the header should
        no longer be used.
        """
        self.set(None, None, units=None)

    def to_header(self):
        if self.units is None:
            return ''
        if self.length is None:
            length = '*'
        else:
            length = self.length
        if self.start is None:
            return '%s */%s' % (self.units, length)
        return '%s %s-%s/%s' % (
            self.units,
            self.start,
            self.stop - 1,
            length
        )

    def __nonzero__(self):
        return self.units is not None

    __bool__ = __nonzero__

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, str(self))


class Authorization(ImmutableDictMixin, dict):

    """Represents an `Authorization` header sent by the client.  You should
    not create this kind of object yourself but use it when it's returned by
    the `parse_authorization_header` function.

    This object is a dict subclass and can be altered by setting dict items
    but it should be considered immutable as it's returned by the client and
    not meant for modifications.

    .. versionchanged:: 0.5
       This object became immutable.
    """

    def __init__(self, auth_type, data=None):
        dict.__init__(self, data or {})
        self.type = auth_type

    username = property(lambda x: x.get('username'), doc='''
        The username transmitted.  This is set for both basic and digest
        auth all the time.''')
    password = property(lambda x: x.get('password'), doc='''
        When the authentication type is basic this is the password
        transmitted by the client, else `None`.''')
    realm = property(lambda x: x.get('realm'), doc='''
        This is the server realm sent back for HTTP digest auth.''')
    nonce = property(lambda x: x.get('nonce'), doc='''
        The nonce the server sent for digest auth, sent back by the client.
        A nonce should be unique for every 401 response for HTTP digest
        auth.''')
    uri = property(lambda x: x.get('uri'), doc='''
        The URI from Request-URI of the Request-Line; duplicated because
        proxies are allowed to change the Request-Line in transit.  HTTP
        digest auth only.''')
    nc = property(lambda x: x.get('nc'), doc='''
        The nonce count value transmitted by clients if a qop-header is
        also transmitted.  HTTP digest auth only.''')
    cnonce = property(lambda x: x.get('cnonce'), doc='''
        If the server sent a qop-header in the ``WWW-Authenticate``
        header, the client has to provide this value for HTTP digest auth.
        See the RFC for more details.''')
    response = property(lambda x: x.get('response'), doc='''
        A string of 32 hex digits computed as defined in RFC 2617, which
        proves that the user knows a password.  Digest auth only.''')
    opaque = property(lambda x: x.get('opaque'), doc='''
        The opaque header from the server returned unchanged by the client.
        It is recommended that this string be base64 or hexadecimal data.
        Digest auth only.''')

    @property
    def qop(self):
        """Indicates what "quality of protection" the client has applied to
        the message for HTTP digest auth."""
        def on_update(header_set):
            if not header_set and 'qop' in self:
                del self['qop']
            elif header_set:
                self['qop'] = header_set.to_header()
        return parse_set_header(self.get('qop'), on_update)


class WWWAuthenticate(UpdateDictMixin, dict):

    """Provides simple access to `WWW-Authenticate` headers."""

    #: list of keys that require quoting in the generated header
    _require_quoting = frozenset(['domain', 'nonce', 'opaque', 'realm', 'qop'])

    def __init__(self, auth_type=None, values=None, on_update=None):
        dict.__init__(self, values or ())
        if auth_type:
            self['__auth_type__'] = auth_type
        self.on_update = on_update

    def set_basic(self, realm='authentication required'):
        """Clear the auth info and enable basic auth."""
        dict.clear(self)
        dict.update(self, {'__auth_type__': 'basic', 'realm': realm})
        if self.on_update:
            self.on_update(self)

    def set_digest(self, realm, nonce, qop=('auth',), opaque=None,
                   algorithm=None, stale=False):
        """Clear the auth info and enable digest auth."""
        d = {
            '__auth_type__':    'digest',
            'realm':            realm,
            'nonce':            nonce,
            'qop':              dump_header(qop)
        }
        if stale:
            d['stale'] = 'TRUE'
        if opaque is not None:
            d['opaque'] = opaque
        if algorithm is not None:
            d['algorithm'] = algorithm
        dict.clear(self)
        dict.update(self, d)
        if self.on_update:
            self.on_update(self)

    def to_header(self):
        """Convert the stored values into a WWW-Authenticate header."""
        d = dict(self)
        auth_type = d.pop('__auth_type__', None) or 'basic'
        return '%s %s' % (auth_type.title(), ', '.join([
            '%s=%s' % (key, quote_header_value(value,
                                               allow_token=key not in self._require_quoting))
            for key, value in iteritems(d)
        ]))

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return '<%s %r>' % (
            self.__class__.__name__,
            self.to_header()
        )

    def auth_property(name, doc=None):
        """A static helper function for subclasses to add extra authentication
        system properties onto a class::

            class FooAuthenticate(WWWAuthenticate):
                special_realm = auth_property('special_realm')

        For more information have a look at the sourcecode to see how the
        regular properties (:attr:`realm` etc.) are implemented.
        """

        def _set_value(self, value):
            if value is None:
                self.pop(name, None)
            else:
                self[name] = str(value)
        return property(lambda x: x.get(name), _set_value, doc=doc)

    def _set_property(name, doc=None):
        def fget(self):
            def on_update(header_set):
                if not header_set and name in self:
                    del self[name]
                elif header_set:
                    self[name] = header_set.to_header()
            return parse_set_header(self.get(name), on_update)
        return property(fget, doc=doc)

    type = auth_property('__auth_type__', doc='''
        The type of the auth mechanism.  HTTP currently specifies
        `Basic` and `Digest`.''')
    realm = auth_property('realm', doc='''
        A string to be displayed to users so they know which username and
        password to use.  This string should contain at least the name of
        the host performing the authentication and might additionally
        indicate the collection of users who might have access.''')
    domain = _set_property('domain', doc='''
        A list of URIs that define the protection space.  If a URI is an
        absolute path, it is relative to the canonical root URL of the
        server being accessed.''')
    nonce = auth_property('nonce', doc='''
        A server-specified data string which should be uniquely generated
        each time a 401 response is made.  It is recommended that this
        string be base64 or hexadecimal data.''')
    opaque = auth_property('opaque', doc='''
        A string of data, specified by the server, which should be returned
        by the client unchanged in the Authorization header of subsequent
        requests with URIs in the same protection space.  It is recommended
        that this string be base64 or hexadecimal data.''')
    algorithm = auth_property('algorithm', doc='''
        A string indicating a pair of algorithms used to produce the digest
        and a checksum.  If this is not present it is assumed to be "MD5".
        If the algorithm is not understood, the challenge should be ignored
        (and a different one used, if there is more than one).''')
    qop = _set_property('qop', doc='''
        A set of quality-of-privacy directives such as auth and auth-int.''')

    def _get_stale(self):
        val = self.get('stale')
        if val is not None:
            return val.lower() == 'true'

    def _set_stale(self, value):
        if value is None:
            self.pop('stale', None)
        else:
            self['stale'] = value and 'TRUE' or 'FALSE'
    stale = property(_get_stale, _set_stale, doc='''
        A flag, indicating that the previous request from the client was
        rejected because the nonce value was stale.''')
    del _get_stale, _set_stale

    # make auth_property a staticmethod so that subclasses of
    # `WWWAuthenticate` can use it for new properties.
    auth_property = staticmethod(auth_property)
    del _set_property


class FileStorage(object):

    """The :class:`FileStorage` class is a thin wrapper over incoming files.
    It is used by the request object to represent uploaded files.  All the
    attributes of the wrapper stream are proxied by the file storage so
    it's possible to do ``storage.read()`` instead of the long form
    ``storage.stream.read()``.
    """

    def __init__(self, stream=None, filename=None, name=None,
                 content_type=None, content_length=None,
                 headers=None):
        self.name = name
        self.stream = stream or _empty_stream

        # if no filename is provided we can attempt to get the filename
        # from the stream object passed.  There we have to be careful to
        # skip things like <fdopen>, <stderr> etc.  Python marks these
        # special filenames with angular brackets.
        if filename is None:
            filename = getattr(stream, 'name', None)
            s = make_literal_wrapper(filename)
            if filename and filename[0] == s('<') and filename[-1] == s('>'):
                filename = None

            # On Python 3 we want to make sure the filename is always unicode.
            # This might not be if the name attribute is bytes due to the
            # file being opened from the bytes API.
            if not PY2 and isinstance(filename, bytes):
                filename = filename.decode(get_filesystem_encoding(),
                                           'replace')

        self.filename = filename
        if headers is None:
            headers = Headers()
        self.headers = headers
        if content_type is not None:
            headers['Content-Type'] = content_type
        if content_length is not None:
            headers['Content-Length'] = str(content_length)

    def _parse_content_type(self):
        if not hasattr(self, '_parsed_content_type'):
            self._parsed_content_type = \
                parse_options_header(self.content_type)

    @property
    def content_type(self):
        """The content-type sent in the header.  Usually not available"""
        return self.headers.get('content-type')

    @property
    def content_length(self):
        """The content-length sent in the header.  Usually not available"""
        return int(self.headers.get('content-length') or 0)

    @property
    def mimetype(self):
        """Like :attr:`content_type`, but without parameters (eg, without
        charset, type etc.) and always lowercase.  For example if the content
        type is ``text/HTML; charset=utf-8`` the mimetype would be
        ``'text/html'``.

        .. versionadded:: 0.7
        """
        self._parse_content_type()
        return self._parsed_content_type[0].lower()

    @property
    def mimetype_params(self):
        """The mimetype parameters as dict.  For example if the content
        type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.7
        """
        self._parse_content_type()
        return self._parsed_content_type[1]

    def save(self, dst, buffer_size=16384):
        """Save the file to a destination path or file object.  If the
        destination is a file object you have to close it yourself after the
        call.  The buffer size is the number of bytes held in memory during
        the copy process.  It defaults to 16KB.

        For secure file saving also have a look at :func:`secure_filename`.

        :param dst: a filename or open file object the uploaded file
                    is saved to.
        :param buffer_size: the size of the buffer.  This works the same as
                            the `length` parameter of
                            :func:`shutil.copyfileobj`.
        """
        from shutil import copyfileobj
        close_dst = False
        if isinstance(dst, string_types):
            dst = open(dst, 'wb')
            close_dst = True
        try:
            copyfileobj(self.stream, dst, buffer_size)
        finally:
            if close_dst:
                dst.close()

    def close(self):
        """Close the underlying file if possible."""
        try:
            self.stream.close()
        except Exception:
            pass

    def __nonzero__(self):
        return bool(self.filename)
    __bool__ = __nonzero__

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __iter__(self):
        return iter(self.readline, '')

    def __repr__(self):
        return '<%s: %r (%r)>' % (
            self.__class__.__name__,
            self.filename,
            self.content_type
        )


# circular dependencies
from werkzeug.http import dump_options_header, dump_header, generate_etag, \
    quote_header_value, parse_set_header, unquote_etag, quote_etag, \
    parse_options_header, http_date, is_byte_range_valid
from werkzeug import exceptions
