import abc
import collections
import collections.abc
import functools
import operator
import sys
import types as _types
import typing


__all__ = [
    # Super-special typing primitives.
    'Any',
    'ClassVar',
    'Concatenate',
    'Final',
    'LiteralString',
    'ParamSpec',
    'ParamSpecArgs',
    'ParamSpecKwargs',
    'Self',
    'Type',
    'TypeVar',
    'TypeVarTuple',
    'Unpack',

    # ABCs (from collections.abc).
    'Awaitable',
    'AsyncIterator',
    'AsyncIterable',
    'Coroutine',
    'AsyncGenerator',
    'AsyncContextManager',
    'ChainMap',

    # Concrete collection types.
    'ContextManager',
    'Counter',
    'Deque',
    'DefaultDict',
    'NamedTuple',
    'OrderedDict',
    'TypedDict',

    # Structural checks, a.k.a. protocols.
    'SupportsIndex',

    # One-off things.
    'Annotated',
    'assert_never',
    'assert_type',
    'clear_overloads',
    'dataclass_transform',
    'get_overloads',
    'final',
    'get_args',
    'get_origin',
    'get_type_hints',
    'IntVar',
    'is_typeddict',
    'Literal',
    'NewType',
    'overload',
    'override',
    'Protocol',
    'reveal_type',
    'runtime',
    'runtime_checkable',
    'Text',
    'TypeAlias',
    'TypeGuard',
    'TYPE_CHECKING',
    'Never',
    'NoReturn',
    'Required',
    'NotRequired',
]

# for backward compatibility
PEP_560 = True
GenericMeta = type

# The functions below are modified copies of typing internal helpers.
# They are needed by _ProtocolMeta and they provide support for PEP 646.

_marker = object()


def _check_generic(cls, parameters, elen=_marker):
    """Check correct count for parameters of a generic cls (internal helper).
    This gives a nice error message in case of count mismatch.
    """
    if not elen:
        raise TypeError(f"{cls} is not a generic class")
    if elen is _marker:
        if not hasattr(cls, "__parameters__") or not cls.__parameters__:
            raise TypeError(f"{cls} is not a generic class")
        elen = len(cls.__parameters__)
    alen = len(parameters)
    if alen != elen:
        if hasattr(cls, "__parameters__"):
            parameters = [p for p in cls.__parameters__ if not _is_unpack(p)]
            num_tv_tuples = sum(isinstance(p, TypeVarTuple) for p in parameters)
            if (num_tv_tuples > 0) and (alen >= elen - num_tv_tuples):
                return
        raise TypeError(f"Too {'many' if alen > elen else 'few'} parameters for {cls};"
                        f" actual {alen}, expected {elen}")


if sys.version_info >= (3, 10):
    def _should_collect_from_parameters(t):
        return isinstance(
            t, (typing._GenericAlias, _types.GenericAlias, _types.UnionType)
        )
elif sys.version_info >= (3, 9):
    def _should_collect_from_parameters(t):
        return isinstance(t, (typing._GenericAlias, _types.GenericAlias))
else:
    def _should_collect_from_parameters(t):
        return isinstance(t, typing._GenericAlias) and not t._special


def _collect_type_vars(types, typevar_types=None):
    """Collect all type variable contained in types in order of
    first appearance (lexicographic order). For example::

        _collect_type_vars((T, List[S, T])) == (T, S)
    """
    if typevar_types is None:
        typevar_types = typing.TypeVar
    tvars = []
    for t in types:
        if (
            isinstance(t, typevar_types) and
            t not in tvars and
            not _is_unpack(t)
        ):
            tvars.append(t)
        if _should_collect_from_parameters(t):
            tvars.extend([t for t in t.__parameters__ if t not in tvars])
    return tuple(tvars)


NoReturn = typing.NoReturn

# Some unconstrained type variables.  These are used by the container types.
# (These are not for export.)
T = typing.TypeVar('T')  # Any type.
KT = typing.TypeVar('KT')  # Key type.
VT = typing.TypeVar('VT')  # Value type.
T_co = typing.TypeVar('T_co', covariant=True)  # Any type covariant containers.
T_contra = typing.TypeVar('T_contra', contravariant=True)  # Ditto contravariant.


if sys.version_info >= (3, 11):
    from typing import Any
else:

    class _AnyMeta(type):
        def __instancecheck__(self, obj):
            if self is Any:
                raise TypeError("typing_extensions.Any cannot be used with isinstance()")
            return super().__instancecheck__(obj)

        def __repr__(self):
            if self is Any:
                return "typing_extensions.Any"
            return super().__repr__()

    class Any(metaclass=_AnyMeta):
        """Special type indicating an unconstrained type.
        - Any is compatible with every type.
        - Any assumed to have all methods.
        - All values assumed to be instances of Any.
        Note that all the above statements are true from the point of view of
        static type checkers. At runtime, Any should not be used with instance
        checks.
        """
        def __new__(cls, *args, **kwargs):
            if cls is Any:
                raise TypeError("Any cannot be instantiated")
            return super().__new__(cls, *args, **kwargs)


ClassVar = typing.ClassVar

# On older versions of typing there is an internal class named "Final".
# 3.8+
if hasattr(typing, 'Final') and sys.version_info[:2] >= (3, 7):
    Final = typing.Final
# 3.7
else:
    class _FinalForm(typing._SpecialForm, _root=True):

        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only a single type.')
            return typing._GenericAlias(self, (item,))

    Final = _FinalForm('Final',
                       doc="""A special typing construct to indicate that a name
                       cannot be re-assigned or overridden in a subclass.
                       For example:

                           MAX_SIZE: Final = 9000
                           MAX_SIZE += 1  # Error reported by type checker

                           class Connection:
                               TIMEOUT: Final[int] = 10
                           class FastConnector(Connection):
                               TIMEOUT = 1  # Error reported by type checker

                       There is no runtime checking of these properties.""")

if sys.version_info >= (3, 11):
    final = typing.final
else:
    # @final exists in 3.8+, but we backport it for all versions
    # before 3.11 to keep support for the __final__ attribute.
    # See https://bugs.python.org/issue46342
    def final(f):
        """This decorator can be used to indicate to type checkers that
        the decorated method cannot be overridden, and decorated class
        cannot be subclassed. For example:

            class Base:
                @final
                def done(self) -> None:
                    ...
            class Sub(Base):
                def done(self) -> None:  # Error reported by type checker
                    ...
            @final
            class Leaf:
                ...
            class Other(Leaf):  # Error reported by type checker
                ...

        There is no runtime checking of these properties. The decorator
        sets the ``__final__`` attribute to ``True`` on the decorated object
        to allow runtime introspection.
        """
        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            # Skip the attribute silently if it is not writable.
            # AttributeError happens if the object has __slots__ or a
            # read-only property, TypeError if it's a builtin class.
            pass
        return f


def IntVar(name):
    return typing.TypeVar(name)


# 3.8+:
if hasattr(typing, 'Literal'):
    Literal = typing.Literal
# 3.7:
else:
    class _LiteralForm(typing._SpecialForm, _root=True):

        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            return typing._GenericAlias(self, parameters)

    Literal = _LiteralForm('Literal',
                           doc="""A type that can be used to indicate to type checkers
                           that the corresponding value has a value literally equivalent
                           to the provided parameter. For example:

                               var: Literal[4] = 4

                           The type checker understands that 'var' is literally equal to
                           the value 4 and no other value.

                           Literal[...] cannot be subclassed. There is no runtime
                           checking verifying that the parameter is actually a value
                           instead of a type.""")


_overload_dummy = typing._overload_dummy  # noqa


if hasattr(typing, "get_overloads"):  # 3.11+
    overload = typing.overload
    get_overloads = typing.get_overloads
    clear_overloads = typing.clear_overloads
else:
    # {module: {qualname: {firstlineno: func}}}
    _overload_registry = collections.defaultdict(
        functools.partial(collections.defaultdict, dict)
    )

    def overload(func):
        """Decorator for overloaded functions/methods.

        In a stub file, place two or more stub definitions for the same
        function in a row, each decorated with @overload.  For example:

        @overload
        def utf8(value: None) -> None: ...
        @overload
        def utf8(value: bytes) -> bytes: ...
        @overload
        def utf8(value: str) -> bytes: ...

        In a non-stub file (i.e. a regular .py file), do the same but
        follow it with an implementation.  The implementation should *not*
        be decorated with @overload.  For example:

        @overload
        def utf8(value: None) -> None: ...
        @overload
        def utf8(value: bytes) -> bytes: ...
        @overload
        def utf8(value: str) -> bytes: ...
        def utf8(value):
            # implementation goes here

        The overloads for a function can be retrieved at runtime using the
        get_overloads() function.
        """
        # classmethod and staticmethod
        f = getattr(func, "__func__", func)
        try:
            _overload_registry[f.__module__][f.__qualname__][
                f.__code__.co_firstlineno
            ] = func
        except AttributeError:
            # Not a normal function; ignore.
            pass
        return _overload_dummy

    def get_overloads(func):
        """Return all defined overloads for *func* as a sequence."""
        # classmethod and staticmethod
        f = getattr(func, "__func__", func)
        if f.__module__ not in _overload_registry:
            return []
        mod_dict = _overload_registry[f.__module__]
        if f.__qualname__ not in mod_dict:
            return []
        return list(mod_dict[f.__qualname__].values())

    def clear_overloads():
        """Clear all overloads in the registry."""
        _overload_registry.clear()


# This is not a real generic class.  Don't use outside annotations.
Type = typing.Type

# Various ABCs mimicking those in collections.abc.
# A few are simply re-exported for completeness.


Awaitable = typing.Awaitable
Coroutine = typing.Coroutine
AsyncIterable = typing.AsyncIterable
AsyncIterator = typing.AsyncIterator
Deque = typing.Deque
ContextManager = typing.ContextManager
AsyncContextManager = typing.AsyncContextManager
DefaultDict = typing.DefaultDict

# 3.7.2+
if hasattr(typing, 'OrderedDict'):
    OrderedDict = typing.OrderedDict
# 3.7.0-3.7.2
else:
    OrderedDict = typing._alias(collections.OrderedDict, (KT, VT))

Counter = typing.Counter
ChainMap = typing.ChainMap
AsyncGenerator = typing.AsyncGenerator
NewType = typing.NewType
Text = typing.Text
TYPE_CHECKING = typing.TYPE_CHECKING


_PROTO_WHITELIST = ['Callable', 'Awaitable',
                    'Iterable', 'Iterator', 'AsyncIterable', 'AsyncIterator',
                    'Hashable', 'Sized', 'Container', 'Collection', 'Reversible',
                    'ContextManager', 'AsyncContextManager']


def _get_protocol_attrs(cls):
    attrs = set()
    for base in cls.__mro__[:-1]:  # without object
        if base.__name__ in ('Protocol', 'Generic'):
            continue
        annotations = getattr(base, '__annotations__', {})
        for attr in list(base.__dict__.keys()) + list(annotations.keys()):
            if (not attr.startswith('_abc_') and attr not in (
                    '__abstractmethods__', '__annotations__', '__weakref__',
                    '_is_protocol', '_is_runtime_protocol', '__dict__',
                    '__args__', '__slots__',
                    '__next_in_mro__', '__parameters__', '__origin__',
                    '__orig_bases__', '__extra__', '__tree_hash__',
                    '__doc__', '__subclasshook__', '__init__', '__new__',
                    '__module__', '_MutableMapping__marker', '_gorg')):
                attrs.add(attr)
    return attrs


def _is_callable_members_only(cls):
    return all(callable(getattr(cls, attr, None)) for attr in _get_protocol_attrs(cls))


def _maybe_adjust_parameters(cls):
    """Helper function used in Protocol.__init_subclass__ and _TypedDictMeta.__new__.

    The contents of this function are very similar
    to logic found in typing.Generic.__init_subclass__
    on the CPython main branch.
    """
    tvars = []
    if '__orig_bases__' in cls.__dict__:
        tvars = typing._collect_type_vars(cls.__orig_bases__)
        # Look for Generic[T1, ..., Tn] or Protocol[T1, ..., Tn].
        # If found, tvars must be a subset of it.
        # If not found, tvars is it.
        # Also check for and reject plain Generic,
        # and reject multiple Generic[...] and/or Protocol[...].
        gvars = None
        for base in cls.__orig_bases__:
            if (isinstance(base, typing._GenericAlias) and
                    base.__origin__ in (typing.Generic, Protocol)):
                # for error messages
                the_base = base.__origin__.__name__
                if gvars is not None:
                    raise TypeError(
                        "Cannot inherit from Generic[...]"
                        " and/or Protocol[...] multiple types.")
                gvars = base.__parameters__
        if gvars is None:
            gvars = tvars
        else:
            tvarset = set(tvars)
            gvarset = set(gvars)
            if not tvarset <= gvarset:
                s_vars = ', '.join(str(t) for t in tvars if t not in gvarset)
                s_args = ', '.join(str(g) for g in gvars)
                raise TypeError(f"Some type variables ({s_vars}) are"
                                f" not listed in {the_base}[{s_args}]")
            tvars = gvars
    cls.__parameters__ = tuple(tvars)


# 3.8+
if hasattr(typing, 'Protocol'):
    Protocol = typing.Protocol
# 3.7
else:

    def _no_init(self, *args, **kwargs):
        if type(self)._is_protocol:
            raise TypeError('Protocols cannot be instantiated')

    class _ProtocolMeta(abc.ABCMeta):  # noqa: B024
        # This metaclass is a bit unfortunate and exists only because of the lack
        # of __instancehook__.
        def __instancecheck__(cls, instance):
            # We need this method for situations where attributes are
            # assigned in __init__.
            if ((not getattr(cls, '_is_protocol', False) or
                 _is_callable_members_only(cls)) and
                    issubclass(instance.__class__, cls)):
                return True
            if cls._is_protocol:
                if all(hasattr(instance, attr) and
                       (not callable(getattr(cls, attr, None)) or
                        getattr(instance, attr) is not None)
                       for attr in _get_protocol_attrs(cls)):
                    return True
            return super().__instancecheck__(instance)

    class Protocol(metaclass=_ProtocolMeta):
        # There is quite a lot of overlapping code with typing.Generic.
        # Unfortunately it is hard to avoid this while these live in two different
        # modules. The duplicated code will be removed when Protocol is moved to typing.
        """Base class for protocol classes. Protocol classes are defined as::

            class Proto(Protocol):
                def meth(self) -> int:
                    ...

        Such classes are primarily used with static type checkers that recognize
        structural subtyping (static duck-typing), for example::

            class C:
                def meth(self) -> int:
                    return 0

            def func(x: Proto) -> int:
                return x.meth()

            func(C())  # Passes static type check

        See PEP 544 for details. Protocol classes decorated with
        @typing_extensions.runtime act as simple-minded runtime protocol that checks
        only the presence of given attributes, ignoring their type signatures.

        Protocol classes can be generic, they are defined as::

            class GenProto(Protocol[T]):
                def meth(self) -> T:
                    ...
        """
        __slots__ = ()
        _is_protocol = True

        def __new__(cls, *args, **kwds):
            if cls is Protocol:
                raise TypeError("Type Protocol cannot be instantiated; "
                                "it can only be used as a base class")
            return super().__new__(cls)

        @typing._tp_cache
        def __class_getitem__(cls, params):
            if not isinstance(params, tuple):
                params = (params,)
            if not params and cls is not typing.Tuple:
                raise TypeError(
                    f"Parameter list to {cls.__qualname__}[...] cannot be empty")
            msg = "Parameters to generic types must be types."
            params = tuple(typing._type_check(p, msg) for p in params)  # noqa
            if cls is Protocol:
                # Generic can only be subscripted with unique type variables.
                if not all(isinstance(p, typing.TypeVar) for p in params):
                    i = 0
                    while isinstance(params[i], typing.TypeVar):
                        i += 1
                    raise TypeError(
                        "Parameters to Protocol[...] must all be type variables."
                        f" Parameter {i + 1} is {params[i]}")
                if len(set(params)) != len(params):
                    raise TypeError(
                        "Parameters to Protocol[...] must all be unique")
            else:
                # Subscripting a regular Generic subclass.
                _check_generic(cls, params, len(cls.__parameters__))
            return typing._GenericAlias(cls, params)

        def __init_subclass__(cls, *args, **kwargs):
            if '__orig_bases__' in cls.__dict__:
                error = typing.Generic in cls.__orig_bases__
            else:
                error = typing.Generic in cls.__bases__
            if error:
                raise TypeError("Cannot inherit from plain Generic")
            _maybe_adjust_parameters(cls)

            # Determine if this is a protocol or a concrete subclass.
            if not cls.__dict__.get('_is_protocol', None):
                cls._is_protocol = any(b is Protocol for b in cls.__bases__)

            # Set (or override) the protocol subclass hook.
            def _proto_hook(other):
                if not cls.__dict__.get('_is_protocol', None):
                    return NotImplemented
                if not getattr(cls, '_is_runtime_protocol', False):
                    if sys._getframe(2).f_globals['__name__'] in ['abc', 'functools']:
                        return NotImplemented
                    raise TypeError("Instance and class checks can only be used with"
                                    " @runtime protocols")
                if not _is_callable_members_only(cls):
                    if sys._getframe(2).f_globals['__name__'] in ['abc', 'functools']:
                        return NotImplemented
                    raise TypeError("Protocols with non-method members"
                                    " don't support issubclass()")
                if not isinstance(other, type):
                    # Same error as for issubclass(1, int)
                    raise TypeError('issubclass() arg 1 must be a class')
                for attr in _get_protocol_attrs(cls):
                    for base in other.__mro__:
                        if attr in base.__dict__:
                            if base.__dict__[attr] is None:
                                return NotImplemented
                            break
                        annotations = getattr(base, '__annotations__', {})
                        if (isinstance(annotations, typing.Mapping) and
                                attr in annotations and
                                isinstance(other, _ProtocolMeta) and
                                other._is_protocol):
                            break
                    else:
                        return NotImplemented
                return True
            if '__subclasshook__' not in cls.__dict__:
                cls.__subclasshook__ = _proto_hook

            # We have nothing more to do for non-protocols.
            if not cls._is_protocol:
                return

            # Check consistency of bases.
            for base in cls.__bases__:
                if not (base in (object, typing.Generic) or
                        base.__module__ == 'collections.abc' and
                        base.__name__ in _PROTO_WHITELIST or
                        isinstance(base, _ProtocolMeta) and base._is_protocol):
                    raise TypeError('Protocols can only inherit from other'
                                    f' protocols, got {repr(base)}')
            cls.__init__ = _no_init


# 3.8+
if hasattr(typing, 'runtime_checkable'):
    runtime_checkable = typing.runtime_checkable
# 3.7
else:
    def runtime_checkable(cls):
        """Mark a protocol class as a runtime protocol, so that it
        can be used with isinstance() and issubclass(). Raise TypeError
        if applied to a non-protocol class.

        This allows a simple-minded structural check very similar to the
        one-offs in collections.abc such as Hashable.
        """
        if not isinstance(cls, _ProtocolMeta) or not cls._is_protocol:
            raise TypeError('@runtime_checkable can be only applied to protocol classes,'
                            f' got {cls!r}')
        cls._is_runtime_protocol = True
        return cls


# Exists for backwards compatibility.
runtime = runtime_checkable


# 3.8+
if hasattr(typing, 'SupportsIndex'):
    SupportsIndex = typing.SupportsIndex
# 3.7
else:
    @runtime_checkable
    class SupportsIndex(Protocol):
        __slots__ = ()

        @abc.abstractmethod
        def __index__(self) -> int:
            pass


if hasattr(typing, "Required"):
    # The standard library TypedDict in Python 3.8 does not store runtime information
    # about which (if any) keys are optional.  See https://bugs.python.org/issue38834
    # The standard library TypedDict in Python 3.9.0/1 does not honour the "total"
    # keyword with old-style TypedDict().  See https://bugs.python.org/issue42059
    # The standard library TypedDict below Python 3.11 does not store runtime
    # information about optional and required keys when using Required or NotRequired.
    # Generic TypedDicts are also impossible using typing.TypedDict on Python <3.11.
    TypedDict = typing.TypedDict
    _TypedDictMeta = typing._TypedDictMeta
    is_typeddict = typing.is_typeddict
else:
    def _check_fails(cls, other):
        try:
            if sys._getframe(1).f_globals['__name__'] not in ['abc',
                                                              'functools',
                                                              'typing']:
                # Typed dicts are only for static structural subtyping.
                raise TypeError('TypedDict does not support instance and class checks')
        except (AttributeError, ValueError):
            pass
        return False

    def _dict_new(*args, **kwargs):
        if not args:
            raise TypeError('TypedDict.__new__(): not enough arguments')
        _, args = args[0], args[1:]  # allow the "cls" keyword be passed
        return dict(*args, **kwargs)

    _dict_new.__text_signature__ = '($cls, _typename, _fields=None, /, **kwargs)'

    def _typeddict_new(*args, total=True, **kwargs):
        if not args:
            raise TypeError('TypedDict.__new__(): not enough arguments')
        _, args = args[0], args[1:]  # allow the "cls" keyword be passed
        if args:
            typename, args = args[0], args[1:]  # allow the "_typename" keyword be passed
        elif '_typename' in kwargs:
            typename = kwargs.pop('_typename')
            import warnings
            warnings.warn("Passing '_typename' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            raise TypeError("TypedDict.__new__() missing 1 required positional "
                            "argument: '_typename'")
        if args:
            try:
                fields, = args  # allow the "_fields" keyword be passed
            except ValueError:
                raise TypeError('TypedDict.__new__() takes from 2 to 3 '
                                f'positional arguments but {len(args) + 2} '
                                'were given')
        elif '_fields' in kwargs and len(kwargs) == 1:
            fields = kwargs.pop('_fields')
            import warnings
            warnings.warn("Passing '_fields' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            fields = None

        if fields is None:
            fields = kwargs
        elif kwargs:
            raise TypeError("TypedDict takes either a dict or keyword arguments,"
                            " but not both")

        ns = {'__annotations__': dict(fields)}
        try:
            # Setting correct module is necessary to make typed dict classes pickleable.
            ns['__module__'] = sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass

        return _TypedDictMeta(typename, (), ns, total=total)

    _typeddict_new.__text_signature__ = ('($cls, _typename, _fields=None,'
                                         ' /, *, total=True, **kwargs)')

    class _TypedDictMeta(type):
        def __init__(cls, name, bases, ns, total=True):
            super().__init__(name, bases, ns)

        def __new__(cls, name, bases, ns, total=True):
            # Create new typed dict class object.
            # This method is called directly when TypedDict is subclassed,
            # or via _typeddict_new when TypedDict is instantiated. This way
            # TypedDict supports all three syntaxes described in its docstring.
            # Subclasses and instances of TypedDict return actual dictionaries
            # via _dict_new.
            ns['__new__'] = _typeddict_new if name == 'TypedDict' else _dict_new
            # Don't insert typing.Generic into __bases__ here,
            # or Generic.__init_subclass__ will raise TypeError
            # in the super().__new__() call.
            # Instead, monkey-patch __bases__ onto the class after it's been created.
            tp_dict = super().__new__(cls, name, (dict,), ns)

            if any(issubclass(base, typing.Generic) for base in bases):
                tp_dict.__bases__ = (typing.Generic, dict)
                _maybe_adjust_parameters(tp_dict)

            annotations = {}
            own_annotations = ns.get('__annotations__', {})
            msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
            own_annotations = {
                n: typing._type_check(tp, msg) for n, tp in own_annotations.items()
            }
            required_keys = set()
            optional_keys = set()

            for base in bases:
                annotations.update(base.__dict__.get('__annotations__', {}))
                required_keys.update(base.__dict__.get('__required_keys__', ()))
                optional_keys.update(base.__dict__.get('__optional_keys__', ()))

            annotations.update(own_annotations)
            for annotation_key, annotation_type in own_annotations.items():
                annotation_origin = get_origin(annotation_type)
                if annotation_origin is Annotated:
                    annotation_args = get_args(annotation_type)
                    if annotation_args:
                        annotation_type = annotation_args[0]
                        annotation_origin = get_origin(annotation_type)

                if annotation_origin is Required:
                    required_keys.add(annotation_key)
                elif annotation_origin is NotRequired:
                    optional_keys.add(annotation_key)
                elif total:
                    required_keys.add(annotation_key)
                else:
                    optional_keys.add(annotation_key)

            tp_dict.__annotations__ = annotations
            tp_dict.__required_keys__ = frozenset(required_keys)
            tp_dict.__optional_keys__ = frozenset(optional_keys)
            if not hasattr(tp_dict, '__total__'):
                tp_dict.__total__ = total
            return tp_dict

        __instancecheck__ = __subclasscheck__ = _check_fails

    TypedDict = _TypedDictMeta('TypedDict', (dict,), {})
    TypedDict.__module__ = __name__
    TypedDict.__doc__ = \
        """A simple typed name space. At runtime it is equivalent to a plain dict.

        TypedDict creates a dictionary type that expects all of its
        instances to have a certain set of keys, with each key
        associated with a value of a consistent type. This expectation
        is not checked at runtime but is only enforced by type checkers.
        Usage::

            class Point2D(TypedDict):
                x: int
                y: int
                label: str

            a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
            b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

            assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

        The type info can be accessed via the Point2D.__annotations__ dict, and
        the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.
        TypedDict supports two additional equivalent forms::

            Point2D = TypedDict('Point2D', x=int, y=int, label=str)
            Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

        The class syntax is only supported in Python 3.6+, while two other
        syntax forms work for Python 2.7 and 3.2+
        """

    if hasattr(typing, "_TypedDictMeta"):
        _TYPEDDICT_TYPES = (typing._TypedDictMeta, _TypedDictMeta)
    else:
        _TYPEDDICT_TYPES = (_TypedDictMeta,)

    def is_typeddict(tp):
        """Check if an annotation is a TypedDict class

        For example::
            class Film(TypedDict):
                title: str
                year: int

            is_typeddict(Film)  # => True
            is_typeddict(Union[list, str])  # => False
        """
        return isinstance(tp, tuple(_TYPEDDICT_TYPES))


if hasattr(typing, "assert_type"):
    assert_type = typing.assert_type

else:
    def assert_type(__val, __typ):
        """Assert (to the type checker) that the value is of the given type.

        When the type checker encounters a call to assert_type(), it
        emits an error if the value is not of the specified type::

            def greet(name: str) -> None:
                assert_type(name, str)  # ok
                assert_type(name, int)  # type checker error

        At runtime this returns the first argument unchanged and otherwise
        does nothing.
        """
        return __val


if hasattr(typing, "Required"):
    get_type_hints = typing.get_type_hints
else:
    import functools
    import types

    # replaces _strip_annotations()
    def _strip_extras(t):
        """Strips Annotated, Required and NotRequired from a given type."""
        if isinstance(t, _AnnotatedAlias):
            return _strip_extras(t.__origin__)
        if hasattr(t, "__origin__") and t.__origin__ in (Required, NotRequired):
            return _strip_extras(t.__args__[0])
        if isinstance(t, typing._GenericAlias):
            stripped_args = tuple(_strip_extras(a) for a in t.__args__)
            if stripped_args == t.__args__:
                return t
            return t.copy_with(stripped_args)
        if hasattr(types, "GenericAlias") and isinstance(t, types.GenericAlias):
            stripped_args = tuple(_strip_extras(a) for a in t.__args__)
            if stripped_args == t.__args__:
                return t
            return types.GenericAlias(t.__origin__, stripped_args)
        if hasattr(types, "UnionType") and isinstance(t, types.UnionType):
            stripped_args = tuple(_strip_extras(a) for a in t.__args__)
            if stripped_args == t.__args__:
                return t
            return functools.reduce(operator.or_, stripped_args)

        return t

    def get_type_hints(obj, globalns=None, localns=None, include_extras=False):
        """Return type hints for an object.

        This is often the same as obj.__annotations__, but it handles
        forward references encoded as string literals, adds Optional[t] if a
        default value equal to None is set and recursively replaces all
        'Annotated[T, ...]', 'Required[T]' or 'NotRequired[T]' with 'T'
        (unless 'include_extras=True').

        The argument may be a module, class, method, or function. The annotations
        are returned as a dictionary. For classes, annotations include also
        inherited members.

        TypeError is raised if the argument is not of a type that can contain
        annotations, and an empty dictionary is returned if no annotations are
        present.

        BEWARE -- the behavior of globalns and localns is counterintuitive
        (unless you are familiar with how eval() and exec() work).  The
        search order is locals first, then globals.

        - If no dict arguments are passed, an attempt is made to use the
          globals from obj (or the respective module's globals for classes),
          and these are also used as the locals.  If the object does not appear
          to have globals, an empty dictionary is used.

        - If one dict argument is passed, it is used for both globals and
          locals.

        - If two dict arguments are passed, they specify globals and
          locals, respectively.
        """
        if hasattr(typing, "Annotated"):
            hint = typing.get_type_hints(
                obj, globalns=globalns, localns=localns, include_extras=True
            )
        else:
            hint = typing.get_type_hints(obj, globalns=globalns, localns=localns)
        if include_extras:
            return hint
        return {k: _strip_extras(t) for k, t in hint.items()}


# Python 3.9+ has PEP 593 (Annotated)
if hasattr(typing, 'Annotated'):
    Annotated = typing.Annotated
    # Not exported and not a public API, but needed for get_origin() and get_args()
    # to work.
    _AnnotatedAlias = typing._AnnotatedAlias
# 3.7-3.8
else:
    class _AnnotatedAlias(typing._GenericAlias, _root=True):
        """Runtime representation of an annotated type.

        At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
        with extra annotations. The alias behaves like a normal typing alias,
        instantiating is the same as instantiating the underlying type, binding
        it to types is also the same.
        """
        def __init__(self, origin, metadata):
            if isinstance(origin, _AnnotatedAlias):
                metadata = origin.__metadata__ + metadata
                origin = origin.__origin__
            super().__init__(origin, origin)
            self.__metadata__ = metadata

        def copy_with(self, params):
            assert len(params) == 1
            new_type = params[0]
            return _AnnotatedAlias(new_type, self.__metadata__)

        def __repr__(self):
            return (f"typing_extensions.Annotated[{typing._type_repr(self.__origin__)}, "
                    f"{', '.join(repr(a) for a in self.__metadata__)}]")

        def __reduce__(self):
            return operator.getitem, (
                Annotated, (self.__origin__,) + self.__metadata__
            )

        def __eq__(self, other):
            if not isinstance(other, _AnnotatedAlias):
                return NotImplemented
            if self.__origin__ != other.__origin__:
                return False
            return self.__metadata__ == other.__metadata__

        def __hash__(self):
            return hash((self.__origin__, self.__metadata__))

    class Annotated:
        """Add context specific metadata to a type.

        Example: Annotated[int, runtime_check.Unsigned] indicates to the
        hypothetical runtime_check module that this type is an unsigned int.
        Every other consumer of this type can ignore this metadata and treat
        this type as int.

        The first argument to Annotated must be a valid type (and will be in
        the __origin__ field), the remaining arguments are kept as a tuple in
        the __extra__ field.

        Details:

        - It's an error to call `Annotated` with less than two arguments.
        - Nested Annotated are flattened::

            Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

        - Instantiating an annotated type is equivalent to instantiating the
        underlying type::

            Annotated[C, Ann1](5) == C(5)

        - Annotated can be used as a generic type alias::

            Optimized = Annotated[T, runtime.Optimize()]
            Optimized[int] == Annotated[int, runtime.Optimize()]

            OptimizedList = Annotated[List[T], runtime.Optimize()]
            OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
        """

        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            raise TypeError("Type Annotated cannot be instantiated.")

        @typing._tp_cache
        def __class_getitem__(cls, params):
            if not isinstance(params, tuple) or len(params) < 2:
                raise TypeError("Annotated[...] should be used "
                                "with at least two arguments (a type and an "
                                "annotation).")
            allowed_special_forms = (ClassVar, Final)
            if get_origin(params[0]) in allowed_special_forms:
                origin = params[0]
            else:
                msg = "Annotated[t, ...]: t must be a type."
                origin = typing._type_check(params[0], msg)
            metadata = tuple(params[1:])
            return _AnnotatedAlias(origin, metadata)

        def __init_subclass__(cls, *args, **kwargs):
            raise TypeError(
                f"Cannot subclass {cls.__module__}.Annotated"
            )

# Python 3.8 has get_origin() and get_args() but those implementations aren't
# Annotated-aware, so we can't use those. Python 3.9's versions don't support
# ParamSpecArgs and ParamSpecKwargs, so only Python 3.10's versions will do.
if sys.version_info[:2] >= (3, 10):
    get_origin = typing.get_origin
    get_args = typing.get_args
# 3.7-3.9
else:
    try:
        # 3.9+
        from typing import _BaseGenericAlias
    except ImportError:
        _BaseGenericAlias = typing._GenericAlias
    try:
        # 3.9+
        from typing import GenericAlias as _typing_GenericAlias
    except ImportError:
        _typing_GenericAlias = typing._GenericAlias

    def get_origin(tp):
        """Get the unsubscripted version of a type.

        This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar
        and Annotated. Return None for unsupported types. Examples::

            get_origin(Literal[42]) is Literal
            get_origin(int) is None
            get_origin(ClassVar[int]) is ClassVar
            get_origin(Generic) is Generic
            get_origin(Generic[T]) is Generic
            get_origin(Union[T, int]) is Union
            get_origin(List[Tuple[T, T]][int]) == list
            get_origin(P.args) is P
        """
        if isinstance(tp, _AnnotatedAlias):
            return Annotated
        if isinstance(tp, (typing._GenericAlias, _typing_GenericAlias, _BaseGenericAlias,
                           ParamSpecArgs, ParamSpecKwargs)):
            return tp.__origin__
        if tp is typing.Generic:
            return typing.Generic
        return None

    def get_args(tp):
        """Get type arguments with all substitutions performed.

        For unions, basic simplifications used by Union constructor are performed.
        Examples::
            get_args(Dict[str, int]) == (str, int)
            get_args(int) == ()
            get_args(Union[int, Union[T, int], str][int]) == (int, str)
            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
            get_args(Callable[[], T][int]) == ([], int)
        """
        if isinstance(tp, _AnnotatedAlias):
            return (tp.__origin__,) + tp.__metadata__
        if isinstance(tp, (typing._GenericAlias, _typing_GenericAlias)):
            if getattr(tp, "_special", False):
                return ()
            res = tp.__args__
            if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res
        return ()


# 3.10+
if hasattr(typing, 'TypeAlias'):
    TypeAlias = typing.TypeAlias
# 3.9
elif sys.version_info[:2] >= (3, 9):
    class _TypeAliasForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    @_TypeAliasForm
    def TypeAlias(self, parameters):
        """Special marker indicating that an assignment should
        be recognized as a proper type alias definition by type
        checkers.

        For example::

            Predicate: TypeAlias = Callable[..., bool]

        It's invalid when used anywhere except as in the example above.
        """
        raise TypeError(f"{self} is not subscriptable")
# 3.7-3.8
else:
    class _TypeAliasForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    TypeAlias = _TypeAliasForm('TypeAlias',
                               doc="""Special marker indicating that an assignment should
                               be recognized as a proper type alias definition by type
                               checkers.

                               For example::

                                   Predicate: TypeAlias = Callable[..., bool]

                               It's invalid when used anywhere except as in the example
                               above.""")


class _DefaultMixin:
    """Mixin for TypeVarLike defaults."""

    __slots__ = ()

    def __init__(self, default):
        if isinstance(default, (tuple, list)):
            self.__default__ = tuple((typing._type_check(d, "Default must be a type")
                                      for d in default))
        elif default:
            self.__default__ = typing._type_check(default, "Default must be a type")
        else:
            self.__default__ = None


# Add default and infer_variance parameters from PEP 696 and 695
class TypeVar(typing.TypeVar, _DefaultMixin, _root=True):
    """Type variable."""

    __module__ = 'typing'

    def __init__(self, name, *constraints, bound=None,
                 covariant=False, contravariant=False,
                 default=None, infer_variance=False):
        super().__init__(name, *constraints, bound=bound, covariant=covariant,
                         contravariant=contravariant)
        _DefaultMixin.__init__(self, default)
        self.__infer_variance__ = infer_variance

        # for pickling:
        try:
            def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            def_mod = None
        if def_mod != 'typing_extensions':
            self.__module__ = def_mod


# Python 3.10+ has PEP 612
if hasattr(typing, 'ParamSpecArgs'):
    ParamSpecArgs = typing.ParamSpecArgs
    ParamSpecKwargs = typing.ParamSpecKwargs
# 3.7-3.9
else:
    class _Immutable:
        """Mixin to indicate that object should not be copied."""
        __slots__ = ()

        def __copy__(self):
            return self

        def __deepcopy__(self, memo):
            return self

    class ParamSpecArgs(_Immutable):
        """The args for a ParamSpec object.

        Given a ParamSpec object P, P.args is an instance of ParamSpecArgs.

        ParamSpecArgs objects have a reference back to their ParamSpec:

        P.args.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """
        def __init__(self, origin):
            self.__origin__ = origin

        def __repr__(self):
            return f"{self.__origin__.__name__}.args"

        def __eq__(self, other):
            if not isinstance(other, ParamSpecArgs):
                return NotImplemented
            return self.__origin__ == other.__origin__

    class ParamSpecKwargs(_Immutable):
        """The kwargs for a ParamSpec object.

        Given a ParamSpec object P, P.kwargs is an instance of ParamSpecKwargs.

        ParamSpecKwargs objects have a reference back to their ParamSpec:

        P.kwargs.__origin__ is P

        This type is meant for runtime introspection and has no special meaning to
        static type checkers.
        """
        def __init__(self, origin):
            self.__origin__ = origin

        def __repr__(self):
            return f"{self.__origin__.__name__}.kwargs"

        def __eq__(self, other):
            if not isinstance(other, ParamSpecKwargs):
                return NotImplemented
            return self.__origin__ == other.__origin__

# 3.10+
if hasattr(typing, 'ParamSpec'):

    # Add default Parameter - PEP 696
    class ParamSpec(typing.ParamSpec, _DefaultMixin, _root=True):
        """Parameter specification variable."""

        __module__ = 'typing'

        def __init__(self, name, *, bound=None, covariant=False, contravariant=False,
                     default=None):
            super().__init__(name, bound=bound, covariant=covariant,
                             contravariant=contravariant)
            _DefaultMixin.__init__(self, default)

            # for pickling:
            try:
                def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                def_mod = None
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

# 3.7-3.9
else:

    # Inherits from list as a workaround for Callable checks in Python < 3.9.2.
    class ParamSpec(list, _DefaultMixin):
        """Parameter specification variable.

        Usage::

           P = ParamSpec('P')

        Parameter specification variables exist primarily for the benefit of static
        type checkers.  They are used to forward the parameter types of one
        callable to another callable, a pattern commonly found in higher order
        functions and decorators.  They are only valid when used in ``Concatenate``,
        or s the first argument to ``Callable``. In Python 3.10 and higher,
        they are also supported in user-defined Generics at runtime.
        See class Generic for more information on generic types.  An
        example for annotating a decorator::

           T = TypeVar('T')
           P = ParamSpec('P')

           def add_logging(f: Callable[P, T]) -> Callable[P, T]:
               '''A type-safe decorator to add logging to a function.'''
               def inner(*args: P.args, **kwargs: P.kwargs) -> T:
                   logging.info(f'{f.__name__} was called')
                   return f(*args, **kwargs)
               return inner

           @add_logging
           def add_two(x: float, y: float) -> float:
               '''Add two numbers together.'''
               return x + y

        Parameter specification variables defined with covariant=True or
        contravariant=True can be used to declare covariant or contravariant
        generic types.  These keyword arguments are valid, but their actual semantics
        are yet to be decided.  See PEP 612 for details.

        Parameter specification variables can be introspected. e.g.:

           P.__name__ == 'T'
           P.__bound__ == None
           P.__covariant__ == False
           P.__contravariant__ == False

        Note that only parameter specification variables defined in global scope can
        be pickled.
        """

        # Trick Generic __parameters__.
        __class__ = typing.TypeVar

        @property
        def args(self):
            return ParamSpecArgs(self)

        @property
        def kwargs(self):
            return ParamSpecKwargs(self)

        def __init__(self, name, *, bound=None, covariant=False, contravariant=False,
                     default=None):
            super().__init__([self])
            self.__name__ = name
            self.__covariant__ = bool(covariant)
            self.__contravariant__ = bool(contravariant)
            if bound:
                self.__bound__ = typing._type_check(bound, 'Bound must be a type.')
            else:
                self.__bound__ = None
            _DefaultMixin.__init__(self, default)

            # for pickling:
            try:
                def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                def_mod = None
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

        def __repr__(self):
            if self.__covariant__:
                prefix = '+'
            elif self.__contravariant__:
                prefix = '-'
            else:
                prefix = '~'
            return prefix + self.__name__

        def __hash__(self):
            return object.__hash__(self)

        def __eq__(self, other):
            return self is other

        def __reduce__(self):
            return self.__name__

        # Hack to get typing._type_check to pass.
        def __call__(self, *args, **kwargs):
            pass


# 3.7-3.9
if not hasattr(typing, 'Concatenate'):
    # Inherits from list as a workaround for Callable checks in Python < 3.9.2.
    class _ConcatenateGenericAlias(list):

        # Trick Generic into looking into this for __parameters__.
        __class__ = typing._GenericAlias

        # Flag in 3.8.
        _special = False

        def __init__(self, origin, args):
            super().__init__(args)
            self.__origin__ = origin
            self.__args__ = args

        def __repr__(self):
            _type_repr = typing._type_repr
            return (f'{_type_repr(self.__origin__)}'
                    f'[{", ".join(_type_repr(arg) for arg in self.__args__)}]')

        def __hash__(self):
            return hash((self.__origin__, self.__args__))

        # Hack to get typing._type_check to pass in Generic.
        def __call__(self, *args, **kwargs):
            pass

        @property
        def __parameters__(self):
            return tuple(
                tp for tp in self.__args__ if isinstance(tp, (typing.TypeVar, ParamSpec))
            )


# 3.7-3.9
@typing._tp_cache
def _concatenate_getitem(self, parameters):
    if parameters == ():
        raise TypeError("Cannot take a Concatenate of no types.")
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    if not isinstance(parameters[-1], ParamSpec):
        raise TypeError("The last parameter to Concatenate should be a "
                        "ParamSpec variable.")
    msg = "Concatenate[arg, ...]: each arg must be a type."
    parameters = tuple(typing._type_check(p, msg) for p in parameters)
    return _ConcatenateGenericAlias(self, parameters)


# 3.10+
if hasattr(typing, 'Concatenate'):
    Concatenate = typing.Concatenate
    _ConcatenateGenericAlias = typing._ConcatenateGenericAlias # noqa
# 3.9
elif sys.version_info[:2] >= (3, 9):
    @_TypeAliasForm
    def Concatenate(self, parameters):
        """Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """
        return _concatenate_getitem(self, parameters)
# 3.7-8
else:
    class _ConcatenateForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            return _concatenate_getitem(self, parameters)

    Concatenate = _ConcatenateForm(
        'Concatenate',
        doc="""Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """)

# 3.10+
if hasattr(typing, 'TypeGuard'):
    TypeGuard = typing.TypeGuard
# 3.9
elif sys.version_info[:2] >= (3, 9):
    class _TypeGuardForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    @_TypeGuardForm
    def TypeGuard(self, parameters):
        """Special typing form used to annotate the return type of a user-defined
        type guard function.  ``TypeGuard`` only accepts a single type argument.
        At runtime, functions marked this way should return a boolean.

        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
        type checkers to determine a more precise type of an expression within a
        program's code flow.  Usually type narrowing is done by analyzing
        conditional code flow and applying the narrowing to a block of code.  The
        conditional expression here is sometimes referred to as a "type guard".

        Sometimes it would be convenient to use a user-defined boolean function
        as a type guard.  Such a function should use ``TypeGuard[...]`` as its
        return type to alert static type checkers to this intention.

        Using  ``-> TypeGuard`` tells the static type checker that for a given
        function:

        1. The return value is a boolean.
        2. If the return value is ``True``, the type of its argument
        is the type inside ``TypeGuard``.

        For example::

            def is_str(val: Union[str, float]):
                # "isinstance" type guard
                if isinstance(val, str):
                    # Type of ``val`` is narrowed to ``str``
                    ...
                else:
                    # Else, type of ``val`` is narrowed to ``float``.
                    ...

        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
        form of ``TypeA`` (it can even be a wider form) and this may lead to
        type-unsafe results.  The main reason is to allow for things like
        narrowing ``List[object]`` to ``List[str]`` even though the latter is not
        a subtype of the former, since ``List`` is invariant.  The responsibility of
        writing type-safe type guards is left to the user.

        ``TypeGuard`` also works with type variables.  For more information, see
        PEP 647 (User-Defined Type Guards).
        """
        item = typing._type_check(parameters, f'{self} accepts only a single type.')
        return typing._GenericAlias(self, (item,))
# 3.7-3.8
else:
    class _TypeGuardForm(typing._SpecialForm, _root=True):

        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only a single type')
            return typing._GenericAlias(self, (item,))

    TypeGuard = _TypeGuardForm(
        'TypeGuard',
        doc="""Special typing form used to annotate the return type of a user-defined
        type guard function.  ``TypeGuard`` only accepts a single type argument.
        At runtime, functions marked this way should return a boolean.

        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
        type checkers to determine a more precise type of an expression within a
        program's code flow.  Usually type narrowing is done by analyzing
        conditional code flow and applying the narrowing to a block of code.  The
        conditional expression here is sometimes referred to as a "type guard".

        Sometimes it would be convenient to use a user-defined boolean function
        as a type guard.  Such a function should use ``TypeGuard[...]`` as its
        return type to alert static type checkers to this intention.

        Using  ``-> TypeGuard`` tells the static type checker that for a given
        function:

        1. The return value is a boolean.
        2. If the return value is ``True``, the type of its argument
        is the type inside ``TypeGuard``.

        For example::

            def is_str(val: Union[str, float]):
                # "isinstance" type guard
                if isinstance(val, str):
                    # Type of ``val`` is narrowed to ``str``
                    ...
                else:
                    # Else, type of ``val`` is narrowed to ``float``.
                    ...

        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
        form of ``TypeA`` (it can even be a wider form) and this may lead to
        type-unsafe results.  The main reason is to allow for things like
        narrowing ``List[object]`` to ``List[str]`` even though the latter is not
        a subtype of the former, since ``List`` is invariant.  The responsibility of
        writing type-safe type guards is left to the user.

        ``TypeGuard`` also works with type variables.  For more information, see
        PEP 647 (User-Defined Type Guards).
        """)


# Vendored from cpython typing._SpecialFrom
class _SpecialForm(typing._Final, _root=True):
    __slots__ = ('_name', '__doc__', '_getitem')

    def __init__(self, getitem):
        self._getitem = getitem
        self._name = getitem.__name__
        self.__doc__ = getitem.__doc__

    def __getattr__(self, item):
        if item in {'__name__', '__qualname__'}:
            return self._name

        raise AttributeError(item)

    def __mro_entries__(self, bases):
        raise TypeError(f"Cannot subclass {self!r}")

    def __repr__(self):
        return f'typing_extensions.{self._name}'

    def __reduce__(self):
        return self._name

    def __call__(self, *args, **kwds):
        raise TypeError(f"Cannot instantiate {self!r}")

    def __or__(self, other):
        return typing.Union[self, other]

    def __ror__(self, other):
        return typing.Union[other, self]

    def __instancecheck__(self, obj):
        raise TypeError(f"{self} cannot be used with isinstance()")

    def __subclasscheck__(self, cls):
        raise TypeError(f"{self} cannot be used with issubclass()")

    @typing._tp_cache
    def __getitem__(self, parameters):
        return self._getitem(self, parameters)


if hasattr(typing, "LiteralString"):
    LiteralString = typing.LiteralString
else:
    @_SpecialForm
    def LiteralString(self, params):
        """Represents an arbitrary literal string.

        Example::

          from pip._vendor.typing_extensions import LiteralString

          def query(sql: LiteralString) -> ...:
              ...

          query("SELECT * FROM table")  # ok
          query(f"SELECT * FROM {input()}")  # not ok

        See PEP 675 for details.

        """
        raise TypeError(f"{self} is not subscriptable")


if hasattr(typing, "Self"):
    Self = typing.Self
else:
    @_SpecialForm
    def Self(self, params):
        """Used to spell the type of "self" in classes.

        Example::

          from typing import Self

          class ReturnsSelf:
              def parse(self, data: bytes) -> Self:
                  ...
                  return self

        """

        raise TypeError(f"{self} is not subscriptable")


if hasattr(typing, "Never"):
    Never = typing.Never
else:
    @_SpecialForm
    def Never(self, params):
        """The bottom type, a type that has no members.

        This can be used to define a function that should never be
        called, or a function that never returns::

            from pip._vendor.typing_extensions import Never

            def never_call_me(arg: Never) -> None:
                pass

            def int_or_str(arg: int | str) -> None:
                never_call_me(arg)  # type checker error
                match arg:
                    case int():
                        print("It's an int")
                    case str():
                        print("It's a str")
                    case _:
                        never_call_me(arg)  # ok, arg is of type Never

        """

        raise TypeError(f"{self} is not subscriptable")


if hasattr(typing, 'Required'):
    Required = typing.Required
    NotRequired = typing.NotRequired
elif sys.version_info[:2] >= (3, 9):
    class _ExtensionsSpecialForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    @_ExtensionsSpecialForm
    def Required(self, parameters):
        """A special typing construct to mark a key of a total=False TypedDict
        as required. For example:

            class Movie(TypedDict, total=False):
                title: Required[str]
                year: int

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )

        There is no runtime checking that a required key is actually provided
        when instantiating a related TypedDict.
        """
        item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
        return typing._GenericAlias(self, (item,))

    @_ExtensionsSpecialForm
    def NotRequired(self, parameters):
        """A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """
        item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
        return typing._GenericAlias(self, (item,))

else:
    class _RequiredForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only a single type.')
            return typing._GenericAlias(self, (item,))

    Required = _RequiredForm(
        'Required',
        doc="""A special typing construct to mark a key of a total=False TypedDict
        as required. For example:

            class Movie(TypedDict, total=False):
                title: Required[str]
                year: int

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )

        There is no runtime checking that a required key is actually provided
        when instantiating a related TypedDict.
        """)
    NotRequired = _RequiredForm(
        'NotRequired',
        doc="""A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """)


if hasattr(typing, "Unpack"):  # 3.11+
    Unpack = typing.Unpack
elif sys.version_info[:2] >= (3, 9):
    class _UnpackSpecialForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

    class _UnpackAlias(typing._GenericAlias, _root=True):
        __class__ = typing.TypeVar

    @_UnpackSpecialForm
    def Unpack(self, parameters):
        """A special typing construct to unpack a variadic type. For example:

            Shape = TypeVarTuple('Shape')
            Batch = NewType('Batch', int)

            def add_batch_axis(
                x: Array[Unpack[Shape]]
            ) -> Array[Batch, Unpack[Shape]]: ...

        """
        item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
        return _UnpackAlias(self, (item,))

    def _is_unpack(obj):
        return isinstance(obj, _UnpackAlias)

else:
    class _UnpackAlias(typing._GenericAlias, _root=True):
        __class__ = typing.TypeVar

    class _UnpackForm(typing._SpecialForm, _root=True):
        def __repr__(self):
            return 'typing_extensions.' + self._name

        def __getitem__(self, parameters):
            item = typing._type_check(parameters,
                                      f'{self._name} accepts only a single type.')
            return _UnpackAlias(self, (item,))

    Unpack = _UnpackForm(
        'Unpack',
        doc="""A special typing construct to unpack a variadic type. For example:

            Shape = TypeVarTuple('Shape')
            Batch = NewType('Batch', int)

            def add_batch_axis(
                x: Array[Unpack[Shape]]
            ) -> Array[Batch, Unpack[Shape]]: ...

        """)

    def _is_unpack(obj):
        return isinstance(obj, _UnpackAlias)


if hasattr(typing, "TypeVarTuple"):  # 3.11+

    # Add default Parameter - PEP 696
    class TypeVarTuple(typing.TypeVarTuple, _DefaultMixin, _root=True):
        """Type variable tuple."""

        def __init__(self, name, *, default=None):
            super().__init__(name)
            _DefaultMixin.__init__(self, default)

            # for pickling:
            try:
                def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                def_mod = None
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

else:
    class TypeVarTuple(_DefaultMixin):
        """Type variable tuple.

        Usage::

            Ts = TypeVarTuple('Ts')

        In the same way that a normal type variable is a stand-in for a single
        type such as ``int``, a type variable *tuple* is a stand-in for a *tuple*
        type such as ``Tuple[int, str]``.

        Type variable tuples can be used in ``Generic`` declarations.
        Consider the following example::

            class Array(Generic[*Ts]): ...

        The ``Ts`` type variable tuple here behaves like ``tuple[T1, T2]``,
        where ``T1`` and ``T2`` are type variables. To use these type variables
        as type parameters of ``Array``, we must *unpack* the type variable tuple using
        the star operator: ``*Ts``. The signature of ``Array`` then behaves
        as if we had simply written ``class Array(Generic[T1, T2]): ...``.
        In contrast to ``Generic[T1, T2]``, however, ``Generic[*Shape]`` allows
        us to parameterise the class with an *arbitrary* number of type parameters.

        Type variable tuples can be used anywhere a normal ``TypeVar`` can.
        This includes class definitions, as shown above, as well as function
        signatures and variable annotations::

            class Array(Generic[*Ts]):

                def __init__(self, shape: Tuple[*Ts]):
                    self._shape: Tuple[*Ts] = shape

                def get_shape(self) -> Tuple[*Ts]:
                    return self._shape

            shape = (Height(480), Width(640))
            x: Array[Height, Width] = Array(shape)
            y = abs(x)  # Inferred type is Array[Height, Width]
            z = x + x   #        ...    is Array[Height, Width]
            x.get_shape()  #     ...    is tuple[Height, Width]

        """

        # Trick Generic __parameters__.
        __class__ = typing.TypeVar

        def __iter__(self):
            yield self.__unpacked__

        def __init__(self, name, *, default=None):
            self.__name__ = name
            _DefaultMixin.__init__(self, default)

            # for pickling:
            try:
                def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                def_mod = None
            if def_mod != 'typing_extensions':
                self.__module__ = def_mod

            self.__unpacked__ = Unpack[self]

        def __repr__(self):
            return self.__name__

        def __hash__(self):
            return object.__hash__(self)

        def __eq__(self, other):
            return self is other

        def __reduce__(self):
            return self.__name__

        def __init_subclass__(self, *args, **kwds):
            if '_root' not in kwds:
                raise TypeError("Cannot subclass special typing classes")


if hasattr(typing, "reveal_type"):
    reveal_type = typing.reveal_type
else:
    def reveal_type(__obj: T) -> T:
        """Reveal the inferred type of a variable.

        When a static type checker encounters a call to ``reveal_type()``,
        it will emit the inferred type of the argument::

            x: int = 1
            reveal_type(x)

        Running a static type checker (e.g., ``mypy``) on this example
        will produce output similar to 'Revealed type is "builtins.int"'.

        At runtime, the function prints the runtime type of the
        argument and returns it unchanged.

        """
        print(f"Runtime type is {type(__obj).__name__!r}", file=sys.stderr)
        return __obj


if hasattr(typing, "assert_never"):
    assert_never = typing.assert_never
else:
    def assert_never(__arg: Never) -> Never:
        """Assert to the type checker that a line of code is unreachable.

        Example::

            def int_or_str(arg: int | str) -> None:
                match arg:
                    case int():
                        print("It's an int")
                    case str():
                        print("It's a str")
                    case _:
                        assert_never(arg)

        If a type checker finds that a call to assert_never() is
        reachable, it will emit an error.

        At runtime, this throws an exception when called.

        """
        raise AssertionError("Expected code to be unreachable")


if hasattr(typing, 'dataclass_transform'):
    dataclass_transform = typing.dataclass_transform
else:
    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        field_specifiers: typing.Tuple[
            typing.Union[typing.Type[typing.Any], typing.Callable[..., typing.Any]],
            ...
        ] = (),
        **kwargs: typing.Any,
    ) -> typing.Callable[[T], T]:
        """Decorator that marks a function, class, or metaclass as providing
        dataclass-like behavior.

        Example:

            from pip._vendor.typing_extensions import dataclass_transform

            _T = TypeVar("_T")

            # Used on a decorator function
            @dataclass_transform()
            def create_model(cls: type[_T]) -> type[_T]:
                ...
                return cls

            @create_model
            class CustomerModel:
                id: int
                name: str

            # Used on a base class
            @dataclass_transform()
            class ModelBase: ...

            class CustomerModel(ModelBase):
                id: int
                name: str

            # Used on a metaclass
            @dataclass_transform()
            class ModelMeta(type): ...

            class ModelBase(metaclass=ModelMeta): ...

            class CustomerModel(ModelBase):
                id: int
                name: str

        Each of the ``CustomerModel`` classes defined in this example will now
        behave similarly to a dataclass created with the ``@dataclasses.dataclass``
        decorator. For example, the type checker will synthesize an ``__init__``
        method.

        The arguments to this decorator can be used to customize this behavior:
        - ``eq_default`` indicates whether the ``eq`` parameter is assumed to be
          True or False if it is omitted by the caller.
        - ``order_default`` indicates whether the ``order`` parameter is
          assumed to be True or False if it is omitted by the caller.
        - ``kw_only_default`` indicates whether the ``kw_only`` parameter is
          assumed to be True or False if it is omitted by the caller.
        - ``field_specifiers`` specifies a static list of supported classes
          or functions that describe fields, similar to ``dataclasses.field()``.

        At runtime, this decorator records its arguments in the
        ``__dataclass_transform__`` attribute on the decorated object.

        See PEP 681 for details.

        """
        def decorator(cls_or_fn):
            cls_or_fn.__dataclass_transform__ = {
                "eq_default": eq_default,
                "order_default": order_default,
                "kw_only_default": kw_only_default,
                "field_specifiers": field_specifiers,
                "kwargs": kwargs,
            }
            return cls_or_fn
        return decorator


if hasattr(typing, "override"):
    override = typing.override
else:
    _F = typing.TypeVar("_F", bound=typing.Callable[..., typing.Any])

    def override(__arg: _F) -> _F:
        """Indicate that a method is intended to override a method in a base class.

        Usage:

            class Base:
                def method(self) -> None: ...
                    pass

            class Child(Base):
                @override
                def method(self) -> None:
                    super().method()

        When this decorator is applied to a method, the type checker will
        validate that it overrides a method with the same name on a base class.
        This helps prevent bugs that may occur when a base class is changed
        without an equivalent change to a child class.

        See PEP 698 for details.

        """
        return __arg


# We have to do some monkey patching to deal with the dual nature of
# Unpack/TypeVarTuple:
# - We want Unpack to be a kind of TypeVar so it gets accepted in
#   Generic[Unpack[Ts]]
# - We want it to *not* be treated as a TypeVar for the purposes of
#   counting generic parameters, so that when we subscript a generic,
#   the runtime doesn't try to substitute the Unpack with the subscripted type.
if not hasattr(typing, "TypeVarTuple"):
    typing._collect_type_vars = _collect_type_vars
    typing._check_generic = _check_generic


# Backport typing.NamedTuple as it exists in Python 3.11.
# In 3.11, the ability to define generic `NamedTuple`s was supported.
# This was explicitly disallowed in 3.9-3.10, and only half-worked in <=3.8.
if sys.version_info >= (3, 11):
    NamedTuple = typing.NamedTuple
else:
    def _caller():
        try:
            return sys._getframe(2).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):  # For platforms without _getframe()
            return None

    def _make_nmtuple(name, types, module, defaults=()):
        fields = [n for n, t in types]
        annotations = {n: typing._type_check(t, f"field {n} annotation must be a type")
                       for n, t in types}
        nm_tpl = collections.namedtuple(name, fields,
                                        defaults=defaults, module=module)
        nm_tpl.__annotations__ = nm_tpl.__new__.__annotations__ = annotations
        # The `_field_types` attribute was removed in 3.9;
        # in earlier versions, it is the same as the `__annotations__` attribute
        if sys.version_info < (3, 9):
            nm_tpl._field_types = annotations
        return nm_tpl

    _prohibited_namedtuple_fields = typing._prohibited
    _special_namedtuple_fields = frozenset({'__module__', '__name__', '__annotations__'})

    class _NamedTupleMeta(type):
        def __new__(cls, typename, bases, ns):
            assert _NamedTuple in bases
            for base in bases:
                if base is not _NamedTuple and base is not typing.Generic:
                    raise TypeError(
                        'can only inherit from a NamedTuple type and Generic')
            bases = tuple(tuple if base is _NamedTuple else base for base in bases)
            types = ns.get('__annotations__', {})
            default_names = []
            for field_name in types:
                if field_name in ns:
                    default_names.append(field_name)
                elif default_names:
                    raise TypeError(f"Non-default namedtuple field {field_name} "
                                    f"cannot follow default field"
                                    f"{'s' if len(default_names) > 1 else ''} "
                                    f"{', '.join(default_names)}")
            nm_tpl = _make_nmtuple(
                typename, types.items(),
                defaults=[ns[n] for n in default_names],
                module=ns['__module__']
            )
            nm_tpl.__bases__ = bases
            if typing.Generic in bases:
                class_getitem = typing.Generic.__class_getitem__.__func__
                nm_tpl.__class_getitem__ = classmethod(class_getitem)
            # update from user namespace without overriding special namedtuple attributes
            for key in ns:
                if key in _prohibited_namedtuple_fields:
                    raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
                elif key not in _special_namedtuple_fields and key not in nm_tpl._fields:
                    setattr(nm_tpl, key, ns[key])
            if typing.Generic in bases:
                nm_tpl.__init_subclass__()
            return nm_tpl

    def NamedTuple(__typename, __fields=None, **kwargs):
        if __fields is None:
            __fields = kwargs.items()
        elif kwargs:
            raise TypeError("Either list of fields or keywords"
                            " can be provided to NamedTuple, not both")
        return _make_nmtuple(__typename, __fields, module=_caller())

    NamedTuple.__doc__ = typing.NamedTuple.__doc__
    _NamedTuple = type.__new__(_NamedTupleMeta, 'NamedTuple', (), {})

    # On 3.8+, alter the signature so that it matches typing.NamedTuple.
    # The signature of typing.NamedTuple on >=3.8 is invalid syntax in Python 3.7,
    # so just leave the signature as it is on 3.7.
    if sys.version_info >= (3, 8):
        NamedTuple.__text_signature__ = '(typename, fields=None, /, **kwargs)'

    def _namedtuple_mro_entries(bases):
        assert NamedTuple in bases
        return (_NamedTuple,)

    NamedTuple.__mro_entries__ = _namedtuple_mro_entries
