# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TFDecorator-aware replacements for the inspect module."""
import collections
import functools
import inspect as _inspect

import six

from tensorflow.python.util import tf_decorator


# inspect.signature() is preferred over inspect.getfullargspec() in PY3.
# Note that while it can handle TFDecorators, it will ignore a TFDecorator's
# provided ArgSpec/FullArgSpec and instead return the signature of the
# inner-most function.
def signature(obj, *, follow_wrapped=True):
  """TFDecorator-aware replacement for inspect.signature."""
  return _inspect.signature(
      tf_decorator.unwrap(obj)[1], follow_wrapped=follow_wrapped)


Parameter = _inspect.Parameter
Signature = _inspect.Signature

if hasattr(_inspect, 'ArgSpec'):
  ArgSpec = _inspect.ArgSpec
else:
  ArgSpec = collections.namedtuple(
      'ArgSpec',
      [
          'args',
          'varargs',
          'keywords',
          'defaults',
      ],
  )


if hasattr(_inspect, 'FullArgSpec'):
  FullArgSpec = _inspect.FullArgSpec  # pylint: disable=invalid-name
else:
  FullArgSpec = collections.namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])


def _convert_maybe_argspec_to_fullargspec(argspec):
  if isinstance(argspec, FullArgSpec):
    return argspec
  return FullArgSpec(
      args=argspec.args,
      varargs=argspec.varargs,
      varkw=argspec.keywords,
      defaults=argspec.defaults,
      kwonlyargs=[],
      kwonlydefaults=None,
      annotations={})

if hasattr(_inspect, 'getfullargspec'):
  _getfullargspec = _inspect.getfullargspec  # pylint: disable=invalid-name

  def _getargspec(target):
    """A python3 version of getargspec.

    Calls `getfullargspec` and assigns args, varargs,
    varkw, and defaults to a python 2/3 compatible `ArgSpec`.

    The parameter name 'varkw' is changed to 'keywords' to fit the
    `ArgSpec` struct.

    Args:
      target: the target object to inspect.

    Returns:
      An ArgSpec with args, varargs, keywords, and defaults parameters
      from FullArgSpec.
    """
    fullargspecs = getfullargspec(target)
    argspecs = ArgSpec(
        args=fullargspecs.args,
        varargs=fullargspecs.varargs,
        keywords=fullargspecs.varkw,
        defaults=fullargspecs.defaults,
    )
    return argspecs
else:
  _getargspec = _inspect.getargspec

  def _getfullargspec(target):
    """A python2 version of getfullargspec.

    Args:
      target: the target object to inspect.

    Returns:
      A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.
    """
    return _convert_maybe_argspec_to_fullargspec(getargspec(target))


def currentframe():
  """TFDecorator-aware replacement for inspect.currentframe."""
  return _inspect.stack()[1][0]


def getargspec(obj):
  """TFDecorator-aware replacement for `inspect.getargspec`.

  Note: `getfullargspec` is recommended as the python 2/3 compatible
  replacement for this function.

  Args:
    obj: A function, partial function, or callable object, possibly decorated.

  Returns:
    The `ArgSpec` that describes the signature of the outermost decorator that
    changes the callable's signature, or the `ArgSpec` that describes
    the object if not decorated.

  Raises:
    ValueError: When callable's signature can not be expressed with
      ArgSpec.
    TypeError: For objects of unsupported types.
  """
  if isinstance(obj, functools.partial):
    return _get_argspec_for_partial(obj)

  decorators, target = tf_decorator.unwrap(obj)

  spec = next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), None)
  if spec:
    return spec

  try:
    # Python3 will handle most callables here (not partial).
    return _getargspec(target)
  except TypeError:
    pass

  if isinstance(target, type):
    try:
      return _getargspec(target.__init__)
    except TypeError:
      pass

    try:
      return _getargspec(target.__new__)
    except TypeError:
      pass

  # The `type(target)` ensures that if a class is received we don't return
  # the signature of its __call__ method.
  return _getargspec(type(target).__call__)


def _get_argspec_for_partial(obj):
  """Implements `getargspec` for `functools.partial` objects.

  Args:
    obj: The `functools.partial` object
  Returns:
    An `inspect.ArgSpec`
  Raises:
    ValueError: When callable's signature can not be expressed with
      ArgSpec.
  """
  # When callable is a functools.partial object, we construct its ArgSpec with
  # following strategy:
  # - If callable partial contains default value for positional arguments (ie.
  # object.args), then final ArgSpec doesn't contain those positional arguments.
  # - If callable partial contains default value for keyword arguments (ie.
  # object.keywords), then we merge them with wrapped target. Default values
  # from callable partial takes precedence over those from wrapped target.
  #
  # However, there is a case where it is impossible to construct a valid
  # ArgSpec. Python requires arguments that have no default values must be
  # defined before those with default values. ArgSpec structure is only valid
  # when this presumption holds true because default values are expressed as a
  # tuple of values without keywords and they are always assumed to belong to
  # last K arguments where K is number of default values present.
  #
  # Since functools.partial can give default value to any argument, this
  # presumption may no longer hold in some cases. For example:
  #
  # def func(m, n):
  #   return 2 * m + n
  # partialed = functools.partial(func, m=1)
  #
  # This example will result in m having a default value but n doesn't. This is
  # usually not allowed in Python and can not be expressed in ArgSpec correctly.
  #
  # Thus, we must detect cases like this by finding first argument with default
  # value and ensures all following arguments also have default values. When
  # this is not true, a ValueError is raised.

  n_prune_args = len(obj.args)
  partial_keywords = obj.keywords or {}

  args, varargs, keywords, defaults = getargspec(obj.func)

  # Pruning first n_prune_args arguments.
  args = args[n_prune_args:]

  # Partial function may give default value to any argument, therefore length
  # of default value list must be len(args) to allow each argument to
  # potentially be given a default value.
  no_default = object()
  all_defaults = [no_default] * len(args)

  if defaults:
    all_defaults[-len(defaults):] = defaults

  # Fill in default values provided by partial function in all_defaults.
  for kw, default in six.iteritems(partial_keywords):
    if kw in args:
      idx = args.index(kw)
      all_defaults[idx] = default
    elif not keywords:
      raise ValueError(f'{obj} does not have a **kwargs parameter, but '
                       f'contains an unknown partial keyword {kw}.')

  # Find first argument with default value set.
  first_default = next(
      (idx for idx, x in enumerate(all_defaults) if x is not no_default), None)

  # If no default values are found, return ArgSpec with defaults=None.
  if first_default is None:
    return ArgSpec(args, varargs, keywords, None)

  # Checks if all arguments have default value set after first one.
  invalid_default_values = [
      args[i] for i, j in enumerate(all_defaults)
      if j is no_default and i > first_default
  ]

  if invalid_default_values:
    raise ValueError(f'{obj} has some keyword-only arguments, which are not'
                     f' supported: {invalid_default_values}.')

  return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))


def getfullargspec(obj):
  """TFDecorator-aware replacement for `inspect.getfullargspec`.

  This wrapper emulates `inspect.getfullargspec` in[^)]* Python2.

  Args:
    obj: A callable, possibly decorated.

  Returns:
    The `FullArgSpec` that describes the signature of
    the outermost decorator that changes the callable's signature. If the
    callable is not decorated, `inspect.getfullargspec()` will be called
    directly on the callable.
  """
  decorators, target = tf_decorator.unwrap(obj)

  for d in decorators:
    if d.decorator_argspec is not None:
      return _convert_maybe_argspec_to_fullargspec(d.decorator_argspec)
  return _getfullargspec(target)


def getcallargs(*func_and_positional, **named):
  """TFDecorator-aware replacement for inspect.getcallargs.

  Args:
    *func_and_positional: A callable, possibly decorated, followed by any
      positional arguments that would be passed to `func`.
    **named: The named argument dictionary that would be passed to `func`.

  Returns:
    A dictionary mapping `func`'s named arguments to the values they would
    receive if `func(*positional, **named)` were called.

  `getcallargs` will use the argspec from the outermost decorator that provides
  it. If no attached decorators modify argspec, the final unwrapped target's
  argspec will be used.
  """
  func = func_and_positional[0]
  positional = func_and_positional[1:]
  argspec = getfullargspec(func)
  call_args = named.copy()
  this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)
  if ismethod(func) and this:
    positional = (this,) + positional
  remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
  call_args.update(dict(zip(remaining_positionals, positional)))
  default_count = 0 if not argspec.defaults else len(argspec.defaults)
  if default_count:
    for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
      if arg not in call_args:
        call_args[arg] = value
  if argspec.kwonlydefaults is not None:
    for k, v in argspec.kwonlydefaults.items():
      if k not in call_args:
        call_args[k] = v
  return call_args


def getframeinfo(*args, **kwargs):
  return _inspect.getframeinfo(*args, **kwargs)


def getdoc(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getdoc.

  Args:
    object: An object, possibly decorated.

  Returns:
    The docstring associated with the object.

  The outermost-decorated object is intended to have the most complete
  documentation, so the decorated parameter is not unwrapped.
  """
  return _inspect.getdoc(object)


def getfile(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getfile."""
  unwrapped_object = tf_decorator.unwrap(object)[1]

  # Work around for the case when object is a stack frame
  # and only .pyc files are used. In this case, getfile
  # might return incorrect path. So, we get the path from f_globals
  # instead.
  if (hasattr(unwrapped_object, 'f_globals') and
      '__file__' in unwrapped_object.f_globals):
    return unwrapped_object.f_globals['__file__']
  return _inspect.getfile(unwrapped_object)


def getmembers(object, predicate=None):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getmembers."""
  return _inspect.getmembers(object, predicate)


def getmodule(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getmodule."""
  return _inspect.getmodule(object)


def getmro(cls):
  """TFDecorator-aware replacement for inspect.getmro."""
  return _inspect.getmro(cls)


def getsource(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getsource."""
  return _inspect.getsource(tf_decorator.unwrap(object)[1])


def getsourcefile(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getsourcefile."""
  return _inspect.getsourcefile(tf_decorator.unwrap(object)[1])


def getsourcelines(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getsourcelines."""
  return _inspect.getsourcelines(tf_decorator.unwrap(object)[1])


def isbuiltin(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isbuiltin."""
  return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])


def isclass(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isclass."""
  return _inspect.isclass(tf_decorator.unwrap(object)[1])


def isfunction(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isfunction."""
  return _inspect.isfunction(tf_decorator.unwrap(object)[1])


def isframe(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.isframe(tf_decorator.unwrap(object)[1])


def isgenerator(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isgenerator."""
  return _inspect.isgenerator(tf_decorator.unwrap(object)[1])


def isgeneratorfunction(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isgeneratorfunction."""
  return _inspect.isgeneratorfunction(tf_decorator.unwrap(object)[1])


def ismethod(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.ismethod."""
  return _inspect.ismethod(tf_decorator.unwrap(object)[1])


def isanytargetmethod(object):  # pylint: disable=redefined-builtin
  # pylint: disable=g-doc-args,g-doc-return-or-yield
  """Checks if `object` or a TF Decorator wrapped target contains self or cls.

  This function could be used along with `tf_inspect.getfullargspec` to
  determine if the first argument of `object` argspec is self or cls. If the
  first argument is self or cls, it needs to be excluded from argspec when we
  compare the argspec to the input arguments and, if provided, the tf.function
  input_signature.

  Like `tf_inspect.getfullargspec` and python `inspect.getfullargspec`, it
  does not unwrap python decorators.

  Args:
    obj: An method, function, or functool.partial, possibly decorated by
    TFDecorator.

  Returns:
    A bool indicates if `object` or any target along the chain of TF decorators
    is a method.
  """
  decorators, target = tf_decorator.unwrap(object)
  for decorator in decorators:
    if _inspect.ismethod(decorator.decorated_target):
      return True

  # TODO(b/194845243): Implement the long term solution with inspect.signature.
  # A functools.partial object is not a function or method. But if the wrapped
  # func is a method, the argspec will contain self/cls.
  while isinstance(target, functools.partial):
    target = target.func

  # `target` is a method or an instance with __call__
  return callable(target) and not _inspect.isfunction(target)


def ismodule(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.ismodule(tf_decorator.unwrap(object)[1])


def isroutine(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isroutine."""
  return _inspect.isroutine(tf_decorator.unwrap(object)[1])


def stack(context=1):
  """TFDecorator-aware replacement for inspect.stack."""
  return _inspect.stack(context)[1:]
