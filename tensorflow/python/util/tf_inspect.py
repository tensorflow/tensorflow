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
import inspect as _inspect

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
    object: A callable, possibly decorated.

  Returns:
    The `ArgSpec` that describes the signature of the outermost decorator that
    changes the callable's signature. If the callable is not decorated,
    `inspect.getargspec()` will be called directly on the callable.
  """
  decorators, target = tf_decorator.unwrap(object)
  return next((d.decorator_argspec for d in decorators
               if d.decorator_argspec is not None), _inspect.getargspec(target))


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
  argspec = getargspec(func)
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
