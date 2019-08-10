# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Minimal runtime type checking library.

This module should not be considered public API.
"""
# TODO(ericmc,shoyer): Delete this in favor of using pytype or mypy
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc

# used for register_type_abbreviation and _type_repr below.
_TYPE_ABBREVIATIONS = {}


class Type(object):
  """Base class for type checker types.

  The custom types defined in this module are based on types in the standard
  library's typing module (in Python 3.5):
  https://docs.python.org/3/library/typing.html

  The only difference should be that we use actual instances of Type classes to
  represent custom types rather than the metaclass magic typing uses to create
  new class objects. In practice, all this should mean is that we use
  `List(int)` rather than `List[int]`.

  Custom types should implement __instancecheck__ and inherit from Type. Every
  argument in the constructor must be a type or Type instance, and these
  arguments must be stored as a tuple on the `_types` attribute.
  """

  def __init__(self, *types):
    self._types = types

  def __repr__(self):
    args_repr = ", ".join(repr(t) for t in self._types)
    return "typecheck.%s(%s)" % (type(self).__name__, args_repr)


class _SingleArgumentType(Type):
  """Use this subclass for parametric types that accept only one argument."""

  def __init__(self, tpe):
    super(_SingleArgumentType, self).__init__(tpe)

  @property
  def _type(self):
    tpe, = self._types  # pylint: disable=unbalanced-tuple-unpacking
    return tpe


class _TwoArgumentType(Type):
  """Use this subclass for parametric types that accept two arguments."""

  def __init__(self, first_type, second_type):
    super(_TwoArgumentType, self).__init__(first_type, second_type)


class Union(Type):
  """A sum type.

  A correct type is any of the types provided.
  """

  def __instancecheck__(self, instance):
    return isinstance(instance, self._types)


class Optional(_SingleArgumentType):
  """An optional type.

  A correct type is either the provided type or NoneType.
  """

  def __instancecheck__(self, instance):
    # types.NoneType does not exist in Python 3
    return isinstance(instance, (self._type, type(None)))


class List(_SingleArgumentType):
  """A typed list.

  A correct type is a list where each element has the single provided type.
  """

  def __instancecheck__(self, instance):
    return (isinstance(instance, list) and
            all(isinstance(x, self._type) for x in instance))


class Sequence(_SingleArgumentType):
  """A typed sequence.

  A correct type is a sequence where each element has the single provided type.
  """

  def __instancecheck__(self, instance):
    return (isinstance(instance, collections_abc.Sequence) and
            all(isinstance(x, self._type) for x in instance))


class Collection(_SingleArgumentType):
  """A sized, iterable container.

  A correct type is an iterable and container with known size where each element
  has the single provided type.

  We use this in preference to Iterable because we check each instance of the
  iterable at runtime, and hence need to avoid iterables that could be
  exhausted.
  """

  def __instancecheck__(self, instance):
    return (isinstance(instance, collections_abc.Iterable) and
            isinstance(instance, collections_abc.Sized) and
            isinstance(instance, collections_abc.Container) and
            all(isinstance(x, self._type) for x in instance))


class Tuple(Type):
  """A typed tuple.

  A correct type is a tuple with the correct length where each element has
  the correct type.
  """

  def __instancecheck__(self, instance):
    return (isinstance(instance, tuple) and
            len(instance) == len(self._types) and
            all(isinstance(x, t) for x, t in zip(instance, self._types)))


class Mapping(_TwoArgumentType):
  """A typed mapping.

  A correct type has the correct parametric types for keys and values.
  """

  def __instancecheck__(self, instance):
    key_type, value_type = self._types  # pylint: disable=unbalanced-tuple-unpacking
    return (isinstance(instance, collections_abc.Mapping) and
            all(isinstance(k, key_type) for k in instance.keys()) and
            all(isinstance(k, value_type) for k in instance.values()))


class Dict(Mapping):
  """A typed dict.

  A correct type has the correct parametric types for keys and values.
  """

  def __instancecheck__(self, instance):
    return (isinstance(instance, dict) and
            super(Dict, self).__instancecheck__(instance))


def _replace_forward_references(t, context):
  """Replace forward references in the given type."""
  if isinstance(t, str):
    return context[t]
  elif isinstance(t, Type):
    return type(t)(*[_replace_forward_references(t, context) for t in t._types])  # pylint: disable=protected-access
  else:
    return t


def register_type_abbreviation(name, alias):
  """Register an abbreviation for a type in typecheck tracebacks.

  This makes otherwise very long typecheck errors much more readable.

  Example:
    typecheck.register_type_abbreviation(tf.compat.v1.Dimension,
    'tf.compat.v1.Dimension')

  Args:
    name: type or class to abbreviate.
    alias: string alias to substitute.
  """
  _TYPE_ABBREVIATIONS[name] = alias


def _type_repr(t):
  """A more succinct repr for typecheck tracebacks."""
  string = repr(t)
  for type_, alias in _TYPE_ABBREVIATIONS.items():
    string = string.replace(repr(type_), alias)
  string = re.sub(r"<(class|type) '([\w.]+)'>", r"\2", string)
  string = re.sub(r"typecheck\.(\w+)", r"\1", string)
  return string


class Error(TypeError):
  """Exception for typecheck failures."""


def accepts(*types):
  """A decorator which checks the input types of a function.

  Based on:
  http://stackoverflow.com/questions/15299878/how-to-use-python-decorators-to-check-function-arguments
  The above draws from:
  https://www.python.org/dev/peps/pep-0318/

  Args:
    *types: A list of Python types.

  Returns:
    A function to use as a decorator.
  """

  def check_accepts(f):
    """Check the types."""
    spec = tf_inspect.getargspec(f)

    num_function_arguments = len(spec.args)
    if len(types) != num_function_arguments:
      raise Error(
          "Function %r has %d arguments but only %d types were provided in the "
          "annotation." % (f, num_function_arguments, len(types)))

    if spec.defaults:
      num_defaults = len(spec.defaults)
      for (name, a, t) in zip(spec.args[-num_defaults:], spec.defaults,
                              types[-num_defaults:]):
        allowed_type = _replace_forward_references(t, f.__globals__)
        if not isinstance(a, allowed_type):
          raise Error("default argument value %r of type %r is not an instance "
                      "of the allowed type %s for the %s argument to %r" %
                      (a, type(a), _type_repr(allowed_type), name, f))

    @functools.wraps(f)
    def new_f(*args, **kwds):
      """A helper function."""
      for (a, t) in zip(args, types):
        allowed_type = _replace_forward_references(t, f.__globals__)
        if not isinstance(a, allowed_type):
          raise Error("%r of type %r is not an instance of the allowed type %s "
                      "for %r" % (a, type(a), _type_repr(allowed_type), f))
      return f(*args, **kwds)

    return new_f

  return check_accepts


def returns(*types):
  """A decorator which checks the return types of a function.

  Based on:
  http://stackoverflow.com/questions/15299878/how-to-use-python-decorators-to-check-function-arguments
  The above draws from:
  https://www.python.org/dev/peps/pep-0318/

  Args:
    *types: A list of Python types. A list of one element corresponds to a
      single return value. A list of several elements corresponds to several
      return values. Note that a function with no explicit return value has an
      implicit NoneType return and should be annotated correspondingly.

  Returns:
    A function to use as a decorator.
  """

  def check_returns(f):
    """Check the types."""
    if not types:
      raise TypeError("A return type annotation must contain at least one type")

    @functools.wraps(f)
    def new_f(*args, **kwds):
      """A helper function."""
      return_value = f(*args, **kwds)

      if len(types) == 1:
        # The function has a single return value.
        allowed_type = _replace_forward_references(types[0], f.__globals__)
        if not isinstance(return_value, allowed_type):
          raise Error(
              "%r of type %r is not an instance of the allowed type %s "
              "for %r" %
              (return_value, type(return_value), _type_repr(allowed_type), f))

      else:
        if len(return_value) != len(types):
          raise Error("Function %r has %d return values but only %d types were "
                      "provided in the annotation." %
                      (f, len(return_value), len(types)))

        for (r, t) in zip(return_value, types):
          allowed_type = _replace_forward_references(t, f.__globals__)
          if not isinstance(r, allowed_type):
            raise Error("%r of type %r is not an instance of allowed type %s "
                        "for %r" % (r, type(r), _type_repr(allowed_type), f))

      return return_value

    return new_f

  return check_returns
