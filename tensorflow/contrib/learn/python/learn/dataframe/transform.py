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

"""A Transform takes a list of `Series` and returns a namedtuple of `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import collections
import inspect

from .series import Series
from .series import TransformedSeries


def _make_list_of_series(x):
  """Converts `x` into a list of `Series` if possible.

  Args:
    x: a `Series`, a list of `Series` or `None`.

  Returns:
    `x` if it is a list of Series, `[x]` if `x` is a `Series`, `[]` if x is
    `None`.

  Raises:
    TypeError: `x` is not a `Series` a list of `Series` or `None`.
  """
  if x is None:
    return []
  elif isinstance(x, Series):
    return [x]
  elif isinstance(x, collections.Iterable):
    for i, y in enumerate(x):
      if not isinstance(y, Series):
        raise TypeError(
            "Expected a tuple or list of Series; entry %s has type %s." %
            (i, type(y).__name__))
    return list(x)
  raise TypeError("Expected a Series or list of Series; got %s" %
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
  elif isinstance(x, collections.Iterable):
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
  """A function from a list of `Series` to a namedtuple of `Series`.

  Transforms map zero or more Series of a DataFrame to new Series.
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
    """The number of `Series` that the `Transform` should expect as input.

    `None` indicates that the transform can take a variable number of inputs.

    This function should depend only on `@parameter`s of this `Transform`.

    Returns:
      The number of expected inputs.
    """
    raise NotImplementedError()

  @property
  def output_names(self):
    """The names of `Series` output by the `Transform`.

    This function should depend only on `@parameter`s of this `Transform`.

    Returns:
      A tuple of names of outputs provided by this Transform.
    """
    return _make_tuple_of_string(self._output_names)

  @abstractproperty
  def _output_names(self):
    """The names of `Series` output by the `Transform`.

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
    values are `TransformedSeries`, and for `apply_transform`, in which case
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

  def __str__(self):
    return self.name

  def __repr__(self):
    parameters_sorted = ["%s: %s" % (repr(k), repr(v))
                         for k, v in sorted(self.parameters().items())]
    parameters_joined = ", ".join(parameters_sorted)

    return "%s({%s})" % (self.name, parameters_joined)

  def __call__(self, input_series=None):
    """Apply this `Transform` to the provided `Series`, producing 'Series'.

    Args:
      input_series: None, a `Series`, or a list of input `Series`, acting as
         positional arguments.

    Returns:
      A namedtuple of the output `Series`.

    Raises:
      ValueError: `input_series` does not have expected length
    """
    input_series = _make_list_of_series(input_series)
    if len(input_series) != self.input_valency:
      raise ValueError("Expected %s input Series but received %s." %
                       (self.input_valency, len(input_series)))
    output_series = self._produce_output_series(input_series)

    # pylint: disable=not-callable
    return self.return_type(*output_series)

  @abstractmethod
  def _produce_output_series(self, input_series):
    """Applies the transformation to the `transform_input`.

    Args:
      input_series: a list of Series representing the input to
        the Transform.

    Returns:
        A list of Series representing the transformed output, in order
        corresponding to `_output_names`.
    """
    raise NotImplementedError()


class TensorFlowTransform(Transform):
  """A function from a list of `Series` to a namedtuple of `Series`.

  Transforms map zero or more Series of a DataFrame to new Series.
  """

  __metaclass__ = ABCMeta

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

  def _produce_output_series(self, input_series=None):
    """Apply this `Transform` to the provided `Series`, producing `Series`.

    Args:
      input_series: None, a `Series`, or a list of input `Series`, acting as
         positional arguments.

    Returns:
      A namedtuple of the output `Series`.
    """
    return [TransformedSeries(input_series, self, output_name)
            for output_name in self.output_names]

  def build_transitive(self, input_series, cache=None, **kwargs):
    """Apply this `Transform` to the provided `Series`, producing 'Tensor's.

    Args:
      input_series: None, a `Series`, or a list of input `Series`, acting as
         positional arguments.
      cache: a dict from Series reprs to Tensors.
      **kwargs: Additional keyword arguments, unused here.

    Returns:
      A namedtuple of the output Tensors.

    Raises:
      ValueError: `input_series` does not have expected length
    """
    # pylint: disable=not-callable
    if cache is None:
      cache = {}

    if len(input_series) != self.input_valency:
      raise ValueError("Expected %s input Series but received %s." %
                       (self.input_valency, len(input_series)))
    input_tensors = [series.build(cache, **kwargs) for series in input_series]

    # Note we cache each output individually, not just the entire output
    # tuple.  This allows using the graph as the cache, since it can sensibly
    # cache only individual Tensors.
    output_reprs = [TransformedSeries.make_repr(input_series, self, output_name)
                    for output_name in self.output_names]
    output_tensors = [cache.get(output_repr) for output_repr in output_reprs]

    if None in output_tensors:
      result = self._apply_transform(input_tensors, **kwargs)
      for output_name, output_repr in zip(self.output_names, output_reprs):
        cache[output_repr] = getattr(result, output_name)
    else:
      result = self.return_type(*output_tensors)

    self._check_output_tensors(result)
    return result

  @abstractmethod
  def _apply_transform(self, input_tensors, **kwargs):
    """Applies the transformation to the `transform_input`.

    Args:
      input_tensors: a list of Tensors representing the input to
        the Transform.
      **kwargs: Additional keyword arguments, unused here.

    Returns:
        A namedtuple of Tensors representing the transformed output.
    """
    raise NotImplementedError()
