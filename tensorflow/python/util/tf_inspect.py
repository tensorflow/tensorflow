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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import inspect as _inspect

import six

from tensorflow.python.util import tf_decorator

ArgSpec = _inspect.ArgSpec


if hasattr(_inspect, 'FullArgSpec'):
  FullArgSpec = _inspect.FullArgSpec  # pylint: disable=invalid-name
else:
  FullArgSpec = namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])


def currentframe():
  """TFDecorator-aware replacement for inspect.currentframe."""
  return _inspect.stack()[1][0]


def getargspec(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.getargspec.

  Args:
    object: A callable (function or partial function), possibly decorated.

  Returns:
    The `ArgSpec` that describes the signature of the outermost decorator that
    changes the callable's signature. If the callable is not decorated,
    `inspect.getargspec()` will be called directly on the callable.

  Raises:
    ValueError: When callable's function signature can not be expressed with
    ArgSpec.
  """

  def get_argspec_with_decorator(obj):
    decorators, target = tf_decorator.unwrap(obj)
    return next((d.decorator_argspec
                 for d in decorators
                 if d.decorator_argspec is not None),
                _inspect.getargspec(target))

  if not isinstance(object, functools.partial):
    return get_argspec_with_decorator(object)

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

  n_prune_args = len(object.args)
  partial_keywords = object.keywords or {}

  args, varargs, keywords, defaults = get_argspec_with_decorator(object.func)

  # Pruning first n_prune_args arguments.
  args = args[n_prune_args:]

  # Partial function may give default value to any argument, therefore length
  # of default value list must be len(args) to allow each argument to
  # potentially be given a default value.
  all_defaults = [None] * len(args)
  if defaults:
    all_defaults[-len(defaults):] = defaults

  # Fill in default values provided by partial function in all_defaults.
  for kw, default in six.iteritems(partial_keywords):
    idx = args.index(kw)
    all_defaults[idx] = default

  # Find first argument with default value set.
  first_default = next((idx for idx, x in enumerate(all_defaults) if x), None)

  # If no default values are found, return ArgSpec with defaults=None.
  if first_default is None:
    return ArgSpec(args, varargs, keywords, None)

  # Checks if all arguments have default value set after first one.
  invalid_default_values = [
      args[i] for i, j in enumerate(all_defaults) if not j and i > first_default
  ]

  if invalid_default_values:
    raise ValueError('Some arguments %s do not have default value, but they '
                     'are positioned after those with default values. This can '
                     'not be expressed with ArgSpec.' % invalid_default_values)

  return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))


def getfullargspec(obj):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for `inspect.getfullargspec`/`getargspec`.

  This wrapper uses `inspect.getfullargspec` if available and falls back to
  `inspect.getargspec` in Python 2.

  Args:
    obj: A callable, possibly decorated.

  Returns:
    The `FullArgSpec` that describes the signature of
    the outermost decorator that changes the callable's signature. If the
    callable is not decorated, `inspect.getfullargspec()` will be called
    directly on the callable.
  """
  if hasattr(_inspect, 'getfullargspec'):
    spec_fn = _inspect.getfullargspec
  else:
    def spec_fn(target):
      """Spec function that adding default value from FullArgSpec.

      It is used when getfullargspec is not available (eg in PY2).

      Args:
        target: the target object to inspect.
      Returns:
        The full argument specs with empty kwonlyargs, kwonlydefaults and
        annotations.
      """
      argspecs = _inspect.getargspec(target)
      fullargspecs = FullArgSpec(
          args=argspecs.args,
          varargs=argspecs.varargs,
          varkw=argspecs.keywords,
          defaults=argspecs.defaults,
          kwonlyargs=[],
          kwonlydefaults=None,
          annotations={})
      return fullargspecs

  decorators, target = tf_decorator.unwrap(obj)
  return next((d.decorator_argspec for d in decorators
               if d.decorator_argspec is not None), spec_fn(target))


def getcallargs(func, *positional, **named):
  """TFDecorator-aware replacement for inspect.getcallargs.

  Args:
    func: A callable, possibly decorated
    *positional: The positional arguments that would be passed to `func`.
    **named: The named argument dictionary that would be passed to `func`.

  Returns:
    A dictionary mapping `func`'s named arguments to the values they would
    receive if `func(*positional, **named)` were called.

  `getcallargs` will use the argspec from the outermost decorator that provides
  it. If no attached decorators modify argspec, the final unwrapped target's
  argspec will be used.
  """
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


def isbuiltin(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isbuiltin."""
  return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])


def isclass(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isclass."""
  return _inspect.isclass(tf_decorator.unwrap(object)[1])


def isfunction(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isfunction."""
  return _inspect.isfunction(tf_decorator.unwrap(object)[1])


def ismethod(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.ismethod."""
  return _inspect.ismethod(tf_decorator.unwrap(object)[1])


def ismodule(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.ismodule(tf_decorator.unwrap(object)[1])


def isroutine(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isroutine."""
  return _inspect.isroutine(tf_decorator.unwrap(object)[1])


def stack(context=1):
  """TFDecorator-aware replacement for inspect.stack."""
  return _inspect.stack(context)[1:]
