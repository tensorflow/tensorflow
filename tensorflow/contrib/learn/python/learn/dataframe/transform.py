# pylint: disable=g-bad-file-header
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

"""A Transform takes a list of `Column` and returns a namedtuple of `Column`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import collections
import inspect

from .column import Column
from .column import TransformedColumn


def _make_list_of_column(x):
  """Converts `x` into a list of `Column` if possible.

  Args:
    x: a `Column`, a list of `Column` or `None`.

  Returns:
    `x` if it is a list of Column, `[x]` if `x` is a `Column`, `[]` if x is
    `None`.

  Raises:
    TypeError: `x` is not a `Column` a list of `Column` or `None`.
  """
  if x is None:
    return []
  elif isinstance(x, Column):
    return [x]
  elif isinstance(x, (list, tuple)):
    for i, y in enumerate(x):
      if not isinstance(y, Column):
        raise TypeError(
            "Expected a tuple or list of Columns; entry %s has type %s." %
            (i, type(y).__name__))
    return list(x)
  raise TypeError("Expected a Column or list of Column; got %s" %
                  type(x).__name__)


def _make_tuple_of_string(x):
  """Converts `x` into a list of `str` if possible.

  Args:
    x: a `str`, a list of `str`, a tuple of `str`, or `None`.

  Returns:
    `x` if it is a tuple of str, `tuple(x)` if it is a list of str,
    `(x)` if `x` is a `str`, `()` if x is `None`.

  Raises:
    TypeError: `x` is not a `str`, a list or tuple of `str`, or `None`.
  """
  if x is None:
    return ()
  elif isinstance(x, str):
    return (x,)
  elif isinstance(x, (list, tuple)):
    for i, y in enumerate(x):
      if not isinstance(y, str):
        raise TypeError(
            "Expected a tuple or list of strings; entry %s has type %s." %
            (i, type(y).__name__))
    return x
  raise TypeError("Expected a string or list of strings or tuple of strings; " +
                  "got %s" % type(x).__name__)


def parameter(func):
  """Tag functions annotated with `@parameter` for later retrieval.

  Note that all `@parameter`s are automatically `@property`s as well.

  Args:
    func: the getter function to tag and wrap

  Returns:
    A `@property` whose getter function is marked with is_parameter = True
  """
  func.is_parameter = True
  return property(func)


class Transform(object):
  """A function from a list of `Column` to a namedtuple of `Column`.

  Transforms map zero or more columns of a DataFrame to new columns.
  """

  __metaclass__ = ABCMeta

  def __init__(self):
    self._return_type = None

  @abstractproperty
  def name(self):
    """Name of the transform."""
    raise NotImplementedError()

  def parameters(self):
    """A dict of names to values of properties marked with `@parameter`."""
    property_param_names = [name
                            for name, func in inspect.getmembers(type(self))
                            if (hasattr(func, "fget") and hasattr(
                                getattr(func, "fget"), "is_parameter"))]
    return {name: getattr(self, name) for name in property_param_names}

  @abstractproperty
  def input_valency(self):
    """The number of `Column`s that the `Transform` should expect as input.

    `None` indicates that the transform can take a variable number of inputs.

    This function should depend only on `@parameter`s of this `Transform`.

    Returns:
      The number of expected inputs.
    """
    raise NotImplementedError()

  @property
  def output_names(self):
    """The names of `Column`s output by the `Transform`.

    This function should depend only on `@parameter`s of this `Transform`.

    Returns:
      A tuple of names of outputs provided by this Transform.
    """
    return _make_tuple_of_string(self._output_names)

  @abstractproperty
  def _output_names(self):
    """The names of `Column`s output by the `Transform`.

    This function should depend only on `@parameter`s of this `Transform`.

    Returns:
      Names of outputs provided by this Transform, as a string, tuple, or list.
    """
    raise NotImplementedError()

  @property
  def return_type(self):
    """Provides a namedtuple type which will be used for output.

    A Transform generates one or many outputs, named according to
    _output_names.  This method creates (and caches) a namedtuple type using
    those names as the keys.  The Transform output is then generated by
    instantiating an object of this type with corresponding values.

    Note this output type is used both for `__call__`, in which case the
    values are `TransformedColumn`s, and for `apply_transform`, in which case
    the values are `Tensor`s.

    Returns:
      A namedtuple type fixing the order and names of the outputs of this
        transform.
    """
    if self._return_type is None:
      # TODO(soergel): pylint 3 chokes on this, but it is legit and preferred.
      # return_type_name = "%sReturnType" % type(self).__name__
      return_type_name = "ReturnType"
      self._return_type = collections.namedtuple(return_type_name,
                                                 self.output_names)
    return self._return_type

  def _check_output_tensors(self, output_tensors):
    """Helper for `build(...)`; verifies the output of `_build_transform`.

    Args:
      output_tensors: value returned by a call to `_build_transform`.

    Raises:
      TypeError: `transform_output` is not a list.
      ValueError: `transform_output` does not match `output_names`.
    """
    if not isinstance(output_tensors, self.return_type):
      raise TypeError(
          "Expected a NamedTuple of Tensors with elements %s; got %s." %
          (self.output_names, type(output_tensors).__name__))

  def __call__(self, input_columns=None):
    """Apply this `Transform` to the provided `Column`s, producing 'Column's.

    Args:
      input_columns: None, a `Column`, or a list of input `Column`s, acting as
         positional arguments.

    Returns:
      A namedtuple of the output Columns.

    Raises:
      ValueError: `input_columns` does not have expected length
    """
    input_columns = _make_list_of_column(input_columns)
    if len(input_columns) != self.input_valency:
      raise ValueError("Expected %s input Columns but received %s." %
                       (self.input_valency, len(input_columns)))
    output_columns = [TransformedColumn(input_columns, self, output_name)
                      for output_name in self.output_names]

    # pylint: disable=not-callable
    return self.return_type(*output_columns)

  def apply_transform(self, input_columns, cache=None):
    """Apply this `Transform` to the provided `Column`s, producing 'Tensor's.

    Args:
      input_columns: None, a `Column`, or a list of input `Column`s, acting as
         positional arguments.
      cache: a dict from Column reprs to Tensors.

    Returns:
      A namedtuple of the output Tensors.

    Raises:
      ValueError: `input_columns` does not have expected length
    """
    # pylint: disable=not-callable
    if cache is None:
      cache = {}

    if len(input_columns) != self.input_valency:
      raise ValueError("Expected %s input Columns but received %s." %
                       (self.input_valency, len(input_columns)))
    input_tensors = [input_column.build(cache)
                     for input_column in input_columns]

    # Note we cache each output individually, not just the entire output
    # tuple.  This allows using the graph as the cache, since it can sensibly
    # cache only individual Tensors.
    output_reprs = [TransformedColumn.make_repr(input_columns, self,
                                                output_name)
                    for output_name in self.output_names]
    output_tensors = [cache.get(output_repr) for output_repr in output_reprs]

    if None in output_tensors:
      result = self._apply_transform(input_tensors)
      for output_name, output_repr in zip(self.output_names, output_reprs):
        cache[output_repr] = getattr(result, output_name)
    else:
      result = self.return_type(*output_tensors)

    self._check_output_tensors(result)
    return result

  @abstractmethod
  def _apply_transform(self, input_tensors):
    """Applies the transformation to the `transform_input`.

    Args:
        input_tensors: a list of Tensors representing the input to
        the Transform.

    Returns:
        A namedtuple of Tensors representing the transformed output.
    """
    raise NotImplementedError()

  def __str__(self):
    return self.name

  def __repr__(self):
    parameters_sorted = ["%s: %s" % (repr(k), repr(v))
                         for k, v in sorted(self.parameters().items())]
    parameters_joined = ", ".join(parameters_sorted)

    return "%s({%s})" % (self.name, parameters_joined)
