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
"""Core classes and core ops for LabeledTensor.

Core ops are ops which will eventually be called by LabeledTensor methods,
and ops which a core op depends upon.
For example, `add` is a core op because we'll eventually support the `+`
operator.
Non-core ops should go in `ops.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import numbers
import types

import numpy as np
from six import binary_type
from six import string_types
from six import text_type
from six.moves import range  # pylint: disable=redefined-builtin

from tensorflow.contrib.labeled_tensor.python.ops import _typecheck as tc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# pylint: disable=invalid-name

# Types coercible to Axis.labels
# We use this instead of collections.Sequence to exclude strings.
LabelsLike = tc.Union(np.ndarray, range, list, tuple)

# Types coercible to a tf.Dimension
DimensionLike = tc.Optional(tc.Union(tensor_shape.Dimension, int))

# Types usable for axis values
AxisValue = tc.Union(LabelsLike, DimensionLike)

# Valid scalar values for TensorFlow
Scalar = tc.Union(numbers.Number, bool, binary_type, text_type)

# pylint: enable=invalid-name


class Axis(object):
  """Size and label information for an axis.

  Axis contains either a tf.Dimension indicating the size of an axis,
  or a tuple of tick labels for the axis.

  If tick labels are provided, they must be unique.
  """

  @tc.accepts(object, string_types, AxisValue)
  def __init__(self, name, value):
    """Construct an Axis.

    Args:
      name: Name of the axis.
      value: Either None, an int or tf.Dimension giving the size of the axis,
        or a sequence that is not a string additionally providing coordinate
        (tick) labels.

    Raises:
      ValueError: If the user provides labels with duplicate values.
    """
    if isinstance(value, tensor_shape.Dimension):
      dimension = value
      labels = None
    elif isinstance(value, int) or value is None:
      dimension = tensor_shape.Dimension(value)
      labels = None
    else:
      dimension = tensor_shape.Dimension(len(value))
      labels = tuple(value)

    if dimension.value == 0:
      # Treat a zero-length axis as if it has labels.
      labels = ()

    if labels is not None:
      index = dict(zip(labels, range(len(labels))))
      if len(index) != len(labels):
        raise ValueError('Tick labels must be unique, but got {}'
                         .format(labels))
    else:
      index = None

    self._name = name  # type: string_types
    self._dimension = dimension  # type: tensor_shape.Dimension
    self._labels = labels  # type: Optional[tuple]
    self._index = index  # type: Optional[Dict[Any, int]]

  @property
  @tc.returns(string_types)
  def name(self):
    return self._name

  @tc.returns(string_types)
  def __repr__(self):
    # Axis('x', Dimension(2))
    # TODO(shoyer): make very long reprs more succint?
    return "%s('%s', %r)" % (type(self).__name__, self.name, self.value)

  @tc.returns(bool)
  def __eq__(self, other):
    return (isinstance(other, Axis) and self.name == other.name and
            self.size == other.size and self.labels == other.labels)

  def __hash__(self):
    return hash((self.name, self.size, self.labels))

  @tc.returns(bool)
  def __ne__(self, other):
    return not self == other

  @tc.returns(int)
  def __len__(self):
    size = self.size
    if size is None:
      raise ValueError('axis %r has unknown length' % self.name)
    return size

  @property
  @tc.returns(tc.Optional(tensor_shape.Dimension))
  def dimension(self):
    return self._dimension

  @property
  @tc.returns(tc.Optional(int))
  def size(self):
    return self._dimension.value

  @property
  @tc.returns(tc.Union(tuple, tensor_shape.Dimension))
  def value(self):
    """Returns the tf.Dimension or tuple specifying axis ticks."""
    if self.labels is None:
      return self.dimension
    else:
      return self.labels

  @property
  @tc.returns(tc.Optional(tuple))
  def labels(self):
    """Returns the tuple containing coordinate labels, else None."""
    return self._labels

  def index(self, value):
    """Returns the integer position of the given tick label."""
    if self._index is None:
      raise ValueError('Axis does not have tick labels')
    return self._index[value]


# tc class for anything that can be coerced into an Axis
# pylint: disable=invalid-name
AxisLike = tc.Union(Axis, tc.Tuple(string_types, AxisValue))
# pylint: enable=invalid-name


@tc.returns(Axis)
@tc.accepts(AxisLike)
def as_axis(axis_data):
  """Convert an AxisLike object into an Axis.

  Args:
    axis_data: Axis object or tuple (axis_name, axis_value) describing an axis.

  Returns:
    Axis object. This may be the original object if axis_data is an Axis.
  """
  if isinstance(axis_data, Axis):
    axis = axis_data
  else:
    axis = Axis(*axis_data)
  return axis


class Axes(collections.Mapping):
  """Axis names and indices for a tensor.

  It is an ordered mapping, with keys given by axis name and values given
  by Axis objets. Duplicate axis names are not allowed.
  """

  @tc.accepts(object, tc.List(AxisLike))
  def __init__(self, axes):
    """Construct an Axes.

    Args:
      axes: A list of Axis objects or (axis_name, axis_value) tuples.

    Raises:
      ValueError: If the user provides empty or duplicate axis names.
    """
    self._axes = collections.OrderedDict()

    for axis_data in axes:
      axis = as_axis(axis_data)

      name = axis.name
      if name in self._axes:
        raise ValueError('Duplicate axis name: %s' % name)

      self._axes[name] = axis

  def __iter__(self):
    return iter(self._axes)

  @tc.returns(string_types)
  def __repr__(self):
    # Axes([('x', Dimension(2)),
    #       ('y', ['a', 'b', 'c']),
    #       ('z', Dimension(4))])
    cls_name = type(self).__name__
    values = ["('%s', %r)" % (v.name, v.value) for v in self._axes.values()]
    values_repr = (',\n' + ' ' * len(cls_name + '([')).join(values)
    return '%s([%s])' % (cls_name, values_repr)

  @tc.returns(Axis)
  @tc.accepts(object, string_types)
  def __getitem__(self, name):
    return self._axes[name]

  @tc.returns(bool)
  def __contains__(self, name):
    return name in self._axes

  @tc.returns(int)
  def __len__(self):
    return len(self._axes)

  def __hash__(self):
    return hash(tuple(self.items()))

  @tc.accepts(object, string_types)
  def remove(self, axis_name):
    """Creates a new Axes object without the given axis."""
    if axis_name not in self:
      raise KeyError(axis_name)
    remaining_axes = [axis for axis in self.values() if axis.name != axis_name]
    return Axes(remaining_axes)


class LabeledTensor(object):
  """A tensor with annotated axes.

  It has the following invariants:
    1) The dimensionality of the tensor is equal to the number of elements
    in axes.
    2) The number of coordinate values in the ith dimension is equal to the
    size of the tensor in the ith dimension.

  Attributes:
    tensor: tf.Tensor containing the data.
    axes: lt.Axes containing axis names and coordinate labels.
  """

  @tc.accepts(object, ops.Tensor,
              tc.Union(Axes, tc.Collection(tc.Union(string_types, AxisLike))))
  def __init__(self, tensor, axes):
    """Construct a LabeledTenor.

    Args:
      tensor: The underlying tensor containing the data.
      axes: An Axes object, or a collection of strings, Axis objects or tuples
        of (name, value) pairs indicating the axes.

    Raises:
      ValueError: If the provided axes do not satisfy the class invariants.
    """
    self._tensor = tensor
    shape = tensor.get_shape()

    if isinstance(axes, Axes):
      unvalidated_axes = axes
    else:
      mutable_axes = []

      for position, axis_like in enumerate(axes):
        if isinstance(axis_like, string_types):
          # The coordinates for this axes are unlabeled.
          # Infer the size of the axis.
          value = shape[position]
          axis_like = (axis_like, value)

        mutable_axes.append(axis_like)

      # Construct the Axis object, which will additionally validate the contents
      # of the object.
      unvalidated_axes = Axes(mutable_axes)

    # Check our invariants.

    # First, the rank of the tensor must be equal to the number of axes.
    if len(shape) != len(unvalidated_axes):
      raise ValueError('Tensor rank was not equal to the number of axes: %r, %r'
                       % (shape, unvalidated_axes))

    # Second, the size of each tensor dimension must match the size of the
    # corresponding indices.
    for (d, axis) in zip(shape, unvalidated_axes.values()):
      if d != axis.size:
        raise ValueError(
            'Provided axis size %d does not match tensor dimension size %d' %
            (axis.size, d))

    self._axes = unvalidated_axes

  def __repr__(self):
    # <LabeledTensor 'foo' shape=(2, 3, 4) dtype=float32
    #  axes=[('x', Dimension(2)),
    #        ('y', ('a', 'b', 'c'),
    #        ('z', Dimension(4))]>
    axes = ["('%s', %r)" % (v.name, v.value) for v in self.axes.values()]
    axes_repr = (',\n' + ' ' * len(' axes=[')).join(axes)
    return ("<%s '%s' shape=%s dtype=%s\n axes=[%s]>" %
            (type(self).__name__, self.tensor.name, self.tensor.get_shape(),
             self.tensor.dtype.name, axes_repr))

  @property
  def tensor(self):
    return self._tensor

  def _as_graph_element(self):
    """Support tf.Graph.as_graph_element on LabeledTensor objects.

    This allows operations such as tf.name_scope to take labeled tensors.

    Returns:
      self.tensor
    """
    return self.tensor

  @property
  def axes(self):
    return self._axes

  # properties/methods directly borrowed from tf.Tensor:

  @property
  def dtype(self):
    return self._tensor.dtype

  @property
  def name(self):
    return self._tensor.name

  def get_shape(self):
    """Returns the TensorShape that represents the shape of this tensor.

    See tf.Tensor.get_shape().

    Returns:
      A TensorShape representing the shape of this tensor.
    """
    return self._tensor.get_shape()

  # TODO(shoyer): consider how/if to implement .eval(). Maybe it should return
  # an xarray.DataArray?

  def __getitem__(self, key):
    # This should work exactly like tf.Tensor.__getitem__, except it preserves
    # labels.
    if not isinstance(key, tuple):
      key = (key,)
    if len(key) != len(self.axes):
      raise ValueError('indexer %r must have the same length as the Tensor '
                       'rank (%r)' % (key, len(self.axes)))
    selection = {a: k for a, k in zip(self.axes.keys(), key)}
    return slice_function(self, selection)

  # special methods for overloading arithmetic operations:

  def __abs__(self):
    return abs_function(self)

  def __neg__(self):
    return neg(self)

  def __pos__(self):
    return self

  def __add__(self, other):
    return add(self, other)

  def __radd__(self, other):
    return add(other, self)

  def __sub__(self, other):
    return sub(self, other)

  def __rsub__(self, other):
    return sub(other, self)

  def __mul__(self, other):
    return mul(self, other)

  def __rmul__(self, other):
    return mul(other, self)

  def __truediv__(self, other):
    return div(self, other)

  __div__ = __truediv__

  def __rtruediv__(self, other):
    return div(other, self)

  __rdiv__ = __rtruediv__

  def __mod__(self, other):
    return mod(self, other)

  def __rmod__(self, other):
    return mod(other, self)

  def __pow__(self, other):
    return pow_function(self, other)

  def __rpow__(self, other):
    return pow_function(other, self)

  # logical operations:

  def __invert__(self):
    return logical_not(self)

  def __and__(self, other):
    return logical_and(self, other)

  def __or__(self, other):
    return logical_or(self, other)

  def __xor__(self, other):
    return logical_xor(self, other)

  # boolean operations:

  def __lt__(self, other):
    return less(self, other)

  def __le__(self, other):
    return less_equal(self, other)

  def __gt__(self, other):
    return greater(self, other)

  def __ge__(self, other):
    return greater_equal(self, other)

  def __eq__(self, other):
    # for consistency with tf.Tensor
    if not isinstance(other, LabeledTensor):
      return False

    return self.tensor == other.tensor and self.axes == other.axes

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    return hash((self.tensor, self.axes))


# typecheck type abbreviations:
# abbreviations for third-party types with very long reprs
tc.register_type_abbreviation(tensor_shape.Dimension, 'tensorflow.Dimension')
tc.register_type_abbreviation(ops.Tensor, 'tensorflow.Tensor')
tc.register_type_abbreviation(dtypes.DType, 'tensorflow.DType')
# core LabeledTensor types
tc.register_type_abbreviation(Axis, 'labeled_tensor.Axis')
tc.register_type_abbreviation(Axes, 'labeled_tensor.Axes')
tc.register_type_abbreviation(LabeledTensor, 'labeled_tensor.LabeledTensor')


@tc.returns(ops.Tensor)
@tc.accepts(LabeledTensor)
def _convert_labeled_tensor_to_tensor(value, *args, **kwargs):
  # call ops.convert_to_tensor to handle optional arguments appropriately
  return ops.internal_convert_to_tensor(value.tensor, *args, **kwargs)


ops.register_tensor_conversion_function(LabeledTensor,
                                        _convert_labeled_tensor_to_tensor)

# tc class for anything that can be coerced into a LabeledTensor
# pylint: disable=invalid-name
LabeledTensorLike = tc.Union(LabeledTensor, ops.Tensor, np.ndarray, Scalar)
# pylint: enable=invalid-name


@tc.returns(LabeledTensor)
@tc.accepts(LabeledTensorLike, object, tc.Optional(string_types))
def convert_to_labeled_tensor(value, dtype=None, name=None):
  """Converts the given `value` to a `LabeledTensor`.

  This function accepts `LabeledTensor` objects, 0-dimensional `Tensor` objects
  and numpy arrays, and Python scalars. Higher dimensional unlabeled tensors
  must use the `LabeledTensor` constructor explicitly.

  Args:
    value: Object to convert.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of value.
    name: Optional name to use if a new Tensor is created.

  Returns:
    `value` converted into a `LabeledTensor` object.

  Raises:
    ValueError: If the output would have rank>0 but the input was not already a
      `LabeledTensor`.
  """
  # TODO(shoyer): consider extending to accept xarray.DataArray as input.
  if isinstance(value, LabeledTensor):
    axes = value.axes.values()
    value = value.tensor
  else:
    axes = []

  # We call convert_to_tensor even for LabeledTensor input because it also
  # checks to make sure the dtype argument is compatible.
  tensor = ops.convert_to_tensor(value, dtype=dtype, name=name)
  if len(tensor.get_shape()) != len(axes):
    raise ValueError('cannot automatically convert unlabeled arrays or tensors '
                     'with rank>0 into LabeledTensors: %r' % value)
  return LabeledTensor(tensor, axes)


@tc.returns(Axis)
@tc.accepts(tc.Collection(Axis))
def concat_axes(axes):
  """Concatenate a list of Axes.

  Args:
    axes: A collection of Axis objects.

  Returns:
    The concatenation of the axes.
    If all axes have labels, the result has the concatenation of the labels.
    Else, the result has no labels, and its size is the sum of the sizes
    of the axes.

  Raises:
    ValueError: If `others` is not a collection of Axes or if it is empty.
  """
  if not axes:
    raise ValueError('axes must not be empty')
  for a in axes:
    if not isinstance(a, Axis):
      raise ValueError('Expected an Axis, but got %r of type %r' % (a, type(a)))

  names = set(a.name for a in axes)
  if len(names) > 1:
    raise ValueError('axes do not all have the same name: %r' % names)
  name, = names

  all_have_labels = all(a.labels is not None for a in axes)
  any_has_unknown_size = any(a.size is None for a in axes)

  if all_have_labels:
    value = tuple(label for a in axes for label in a.labels)
  elif any_has_unknown_size:
    value = None
  else:
    value = sum(len(a) for a in axes)
  return Axis(name, value)


@tc.returns(LabeledTensor)
@tc.accepts(LabeledTensorLike, tc.Optional(string_types))
def identity(labeled_tensor, name=None):
  """The identity op.

  See tf.identity.

  Args:
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    The tensor.
  """
  with ops.name_scope(name, 'lt_identity', [labeled_tensor]) as scope:
    labeled_tensor = convert_to_labeled_tensor(labeled_tensor)
    return LabeledTensor(
        array_ops.identity(
            labeled_tensor.tensor, name=scope),
        labeled_tensor.axes)


# We don't call this slice because that shadows a built-in. Instead, we alias
# this to lt.slice in __init__.py.
@tc.returns(LabeledTensor)
@tc.accepts(LabeledTensorLike,
            tc.Mapping(string_types, tc.Union(int, slice)),
            tc.Optional(string_types))
def slice_function(labeled_tensor, selection, name=None):
  """Slice out a subset of the tensor.

  This is an analogue of tf.slice.
  For example:
  >>> tensor = tf.reshape(tf.range(0, 6), [3, 2])
  >>> labeled_tensor = lt.LabeledTensor(tensor, ['a', ('b', ['foo', 'bar'])])
  >>> lt.slice(labeled_tensor, {'a': slice(0, 2), 'b': 1})
  <LabeledTensor 'lt_slice:...' shape=(2,) dtype=int32
   axes=[('a', Dimension(2))]>

  Args:
    labeled_tensor: The input tensor.
    selection: A dictionary of type str -> Union(int, slice of int) mapping
      axis names to sub-selections.
    name: Optional op name.

  Returns:
    The slice as a `LabeledTensor`.
  """
  with ops.name_scope(name, 'lt_slice', [labeled_tensor]) as scope:
    labeled_tensor = convert_to_labeled_tensor(labeled_tensor)

    slices = []

    for axis_name in labeled_tensor.axes:
      if axis_name not in selection:
        # We're not sub-selecting this axis, so use the full slice.
        slices.append(slice(None))
      else:
        slices.append(selection[axis_name])

    sliced_tensor = labeled_tensor.tensor[tuple(slices)]

    sliced_axes = []
    for axis, s in zip(labeled_tensor.axes.values(), slices):
      # We sub-select this axis's index with the slice s.

      # `s` is either an int or a proper slice.
      if isinstance(s, slice):
        if axis.labels is None:
          # We're not tracking coordinate names for this axis.
          sliced_axes.append(axis.name)
        else:
          sliced_axes.append((axis.name, axis.labels[s]))
      else:
        # If the slice is an int this dimension now has size 1, so we remove it.
        assert isinstance(s, int)

    return LabeledTensor(
        array_ops.identity(
            sliced_tensor, name=scope), sliced_axes)


@tc.returns(LabeledTensor)
@tc.accepts(LabeledTensorLike,
            tc.Optional(tc.Collection(string_types)), tc.Optional(string_types))
def transpose(labeled_tensor, axis_order=None, name=None):
  """Permute a tensor's axes.

  See tf.transpose.

  Args:
    labeled_tensor: The input tensor.
    axis_order: Optional desired axis order, as a list of names. By default, the
      order of axes is reversed.
    name: Optional op name.

  Returns:
    The permuted tensor.

  Raises:
    ValueError: If axis_order isn't a permutation of the existing axes.
  """
  with ops.name_scope(name, 'lt_transpose', [labeled_tensor]) as scope:
    labeled_tensor = convert_to_labeled_tensor(labeled_tensor)

    original_order = list(labeled_tensor.axes.keys())
    if axis_order is None:
      axis_order = list(reversed(original_order))
    elif sorted(axis_order) != sorted(original_order):
      raise ValueError(
          'The new axis order must have the same names as the original axes, '
          'but the new order is %r while the original order is %r' %
          (axis_order, original_order))

    axis_names = list(labeled_tensor.axes.keys())
    permutation = [axis_names.index(n) for n in axis_order]

    # Note: TensorFlow doesn't copy data for the identity tranpose.
    transpose_tensor = array_ops.transpose(
        labeled_tensor.tensor, permutation, name=scope)

    permuted_axes = [labeled_tensor.axes[n] for n in axis_order]

    return LabeledTensor(transpose_tensor, permuted_axes)


@tc.returns(LabeledTensor)
@tc.accepts(
    LabeledTensorLike,
    tc.Collection(
        tc.Union(string_types, tc.Tuple(string_types, collections.Hashable))),
    tc.Optional(string_types))
def expand_dims(labeled_tensor, axes, name=None):
  """Insert dimensions of size 1.

  See tf.expand_dims.

  Args:
    labeled_tensor: The input tensor.
    axes: The desired axis names as strings or tuples of (name, label),
      where `label` is the coordinate name for the new dimension `name`.
      These must include the existing axis names, and the existing names must
      appear in the same order in this list as they do in the input tensor.
    name: Optional op name.

  Returns:
    A tensor with an axis for each axis in axes.
    New axes are created with size 1 and do not have labeled coordinates.

  Raises:
    AxisOrderError: If axis names don't appear in the same order in axes
      and the labeled tensor.
  """
  with ops.name_scope(name, 'lt_expand_dims', [labeled_tensor]) as scope:
    labeled_tensor = convert_to_labeled_tensor(labeled_tensor)

    axis_names = [a if isinstance(a, string_types) else a[0] for a in axes]
    check_axis_order(labeled_tensor, axis_names)

    reshaped_axes = []
    shape = []
    for axis_spec in axes:
      if axis_spec in labeled_tensor.axes:
        axis = labeled_tensor.axes[axis_spec]
        reshaped_axes.append(axis)
        shape.append(-1 if axis.size is None else axis.size)
      else:
        if isinstance(axis_spec, string_types):
          reshaped_axes.append((axis_spec, 1))
        else:
          (name, label) = axis_spec
          reshaped_axes.append((name, (label,)))

        shape.append(1)

    reshaped_tensor = array_ops.reshape(
        labeled_tensor.tensor, shape, name=scope)

    return LabeledTensor(reshaped_tensor, reshaped_axes)


# This should only be added to a graph collection once.
_AXIS_ORDER_KEY = ('__axis_order',)


@tc.returns(tc.Optional(tc.List(string_types)))
def get_axis_order():
  """Get the axis_order set by any containing axis_order_scope.

  Returns:
    List of strings giving an order to use for axis names, or None, if no axis
    order is set.
  """
  # By storing axis_order in the graph, we can ensure that axis_order_scope is
  # thread-safe.
  axis_order_list = ops.get_collection(_AXIS_ORDER_KEY)
  if axis_order_list:
    axis_order, = axis_order_list
  else:
    axis_order = None
  return axis_order


@tc.accepts(tc.Optional(tc.List(string_types)))
def _set_axis_order(axis_order):
  axis_order_list = ops.get_collection_ref(_AXIS_ORDER_KEY)
  if axis_order_list:
    axis_order_list[0] = axis_order
  else:
    axis_order_list.append(axis_order)


@contextlib.contextmanager
@tc.accepts(tc.Optional(tc.List(string_types)))
def axis_order_scope(axis_order=None):
  """Set axis order for the result of broadcasting operations within a scope.

  This allows you to ensure that tensors resulting from arithmetic have a
  predictable axis order.

  Example usage:

    with lt.axis_order_scope(['x', 'y', 'z']):
      # result is guaranteed to have the correct axis order
      result = w + b

  You can nest scopes, in which case only the inner-most scope applies, e.g.,

    with lt.axis_order(['x', 'y', 'z']):
      with lt.axis_order():
        result = w + b  # uses the default (left-most) axis ordering

  Args:
    axis_order: optional list of strings providing axis names. By default,
      creates a scope without axis order.

  Yields:
    The provided axis_order or `None`.
  """
  original_axis_order = get_axis_order()
  _set_axis_order(axis_order)
  try:
    yield axis_order
  finally:
    _set_axis_order(original_axis_order)


@tc.returns(tc.List(string_types))
def _get_valid_axis_order():
  axis_order = get_axis_order()
  if axis_order is None:
    raise AxisOrderError('an explicit axis order must be provided with the '
                         'axis_order argument or by using an axis_order_scope')
  return axis_order


class AxisOrderError(ValueError):
  """Error class for cases where there is no valid axis order."""


# TODO(shoyer): should this function accept a list of labeled tensors instead?
@tc.returns(type(None))
@tc.accepts(LabeledTensorLike, tc.Optional(tc.Collection(string_types)))
def check_axis_order(labeled_tensor, axis_order=None):
  """Verify that the given tensor has a consistent axis order.

  Args:
    labeled_tensor: The input tensor. All axes on this tensor must appear in
      axis_order.
    axis_order: Optional desired axis order, as a list of names. If not
      provided, defaults to the current axis_order_scope (if set).

  Raises:
    AxisOrderError: If the axis_order is unavailable, inconsistent or does not
      include all existing axes.
  """
  labeled_tensor = convert_to_labeled_tensor(labeled_tensor)

  if axis_order is None:
    axis_order = _get_valid_axis_order()

  relevant_axis_order = [a for a in axis_order if a in labeled_tensor.axes]

  if len(relevant_axis_order) < len(labeled_tensor.axes):
    raise AxisOrderError(
        'not all axis names appear in the required axis order %r: %r' %
        (axis_order, labeled_tensor))

  if relevant_axis_order != list(labeled_tensor.axes):
    raise AxisOrderError(
        'axes on a labeled tensor do not appear in the same order as the '
        'required axis order %r: %r' % (axis_order, labeled_tensor))


@tc.returns(LabeledTensor)
@tc.accepts(LabeledTensorLike,
            tc.Optional(tc.Collection(string_types)), tc.Optional(string_types))
def impose_axis_order(labeled_tensor, axis_order=None, name=None):
  """Impose desired axis order on a labeled tensor.

  Args:
    labeled_tensor: The input tensor.
    axis_order: Optional desired axis order, as a list of names. If not
      provided, defaults to the current axis_order_scope (if set).
    name: Optional op name.

  Returns:
    Labeled tensor with possibly transposed axes.

  Raises:
    AxisOrderError: If no axis_order is provided or axis_order does not contain
      all axes on the input tensor.
  """
  with ops.name_scope(name, 'lt_impose_axis_order', [labeled_tensor]) as scope:
    labeled_tensor = convert_to_labeled_tensor(labeled_tensor)

    if axis_order is None:
      axis_order = _get_valid_axis_order()

    relevant_axis_order = [a for a in axis_order if a in labeled_tensor.axes]

    return transpose(labeled_tensor, relevant_axis_order, name=scope)


@tc.returns(tc.Optional(list))
@tc.accepts(list, list)
def _find_consistent_ordering(a, b):
  """Find the left-most consistent ordering between two lists of unique items.

  A consistent ordering combines all elements in both a and b while keeping all
  elements in their original order in both inputs. The left-most consistent
  ordering orders elements from `a` not found in `b` before elements in `b` not
  found in `a`.

  For example, given ['x', 'z'] and ['y', 'z'], both ['x', 'y', 'z'] and ['y',
  'x', 'z'] are consistent orderings because each of the inputs appears in
  each consistent ordering in the same order, and ['x', 'y', 'z'] is the
  left-most, because 'x' appears only in `a` and 'y' appears only in `b`. In
  contrast, there is no consistent ordering between ['x', 'y'] and ['y', 'x'].

  Args:
    a: list with unique elements.
    b: list with unique elements.

  Returns:
    List containing all elements in either a or b, or None, if no consistent
    ordering exists.
  """
  a_set = set(a)
  b_set = set(b)
  i = 0
  j = 0
  ordering = []
  while i < len(a) and j < len(b):
    if a[i] not in b_set:
      ordering.append(a[i])
      i += 1
    elif b[j] not in a_set:
      ordering.append(b[j])
      j += 1
    elif a[i] == b[j]:
      ordering.append(a[i])
      i += 1
      j += 1
    else:
      return None

  ordering.extend(a[i:])
  ordering.extend(b[j:])

  return ordering


@tc.returns(LabeledTensor, LabeledTensor, Axes)
@tc.accepts(LabeledTensorLike, LabeledTensorLike, tc.Optional(string_types))
def align(labeled_tensor_0, labeled_tensor_1, name=None):
  """Align the axes of two tensors so they may be broadcast to each other.

  Axes are ordered by the current axis order scope, if present, or by the left-
  most consistent ordering. An exception is raised if it is impossible to align
  the tensors without a transpose (align never copies the input data).

  Example usage:

    >>> a = lt.LabeledTensor(tf.ones((2, 4)), ['x', 'z'])
    >>> b = lt.LabeledTensor(tf.ones((3, 4)), ['y', 'z'])
    >>> a2, b2, axes = lt.align(a, b)
    >>> a2
    <LabeledTensor 'lt_align_1/lt_align_1/0:...' shape=(2, 1, 4) dtype=float32
     axes=[('x', Dimension(2)),
           ('y', Dimension(1)),
           ('z', Dimension(4))]>
    >>> b2
    <LabeledTensor 'lt_align_1/lt_align_1/1:...' shape=(1, 3, 4) dtype=float32
     axes=[('x', Dimension(1)),
           ('y', Dimension(3)),
           ('z', Dimension(4))]>
    >>> axes
    Axes([('x', Dimension(2)),
          ('y', Dimension(3)),
          ('z', Dimension(4))])

  Args:
    labeled_tensor_0: An input tensor.
    labeled_tensor_1: An input tensor.
    name: Optional op name.

  Returns:
    The aligned tensors and the axes the resulting tensor would have if the two
    aligned tensors were broadcast to each other. The aligned tensors have the
    same rank but not necessarily the same shape, with axes in the same order.

  Raises:
    ValueError: If axes with the same name on the inputs are not equal.
    AxisOrderError: If there is no way to reshape the input tensors into the
      output without a transpose.
  """
  with ops.name_scope(name, 'lt_align',
                      [labeled_tensor_0, labeled_tensor_1]) as scope:

    labeled_tensor_0 = convert_to_labeled_tensor(labeled_tensor_0)
    labeled_tensor_1 = convert_to_labeled_tensor(labeled_tensor_1)

    axes_0 = labeled_tensor_0.axes
    axes_1 = labeled_tensor_1.axes
    for axis_name in axes_0:
      if axis_name in axes_1:
        if axes_0[axis_name] != axes_1[axis_name]:
          raise ValueError('Mismatched %r axis on input tensors: %r and %r' %
                           (axis_name, axes_0[axis_name], axes_1[axis_name]))

    axis_scope_order = get_axis_order()
    if axis_scope_order is not None:
      # we are in an axis_order_scope
      axis_names_set = set(axes_0) | set(axes_1)
      new_axis_names = [a for a in axis_scope_order if a in axis_names_set]

      check_axis_order(labeled_tensor_0, axis_scope_order)
      check_axis_order(labeled_tensor_1, axis_scope_order)

    else:
      # attempt to find a consistent ordering
      new_axis_names = _find_consistent_ordering(list(axes_0), list(axes_1))
      if new_axis_names is None:
        raise AxisOrderError(
            'No consistent axis order allows for aligning tensors with axis '
            'orders %r and %r without copying data. Use transpose or '
            'impose_axis_order to reorder axes on one of more of the inputs.' %
            (axes_0.keys(), axes_1.keys()))

    labeled_tensor_0 = expand_dims(
        labeled_tensor_0, new_axis_names, name=scope + '0')
    labeled_tensor_1 = expand_dims(
        labeled_tensor_1, new_axis_names, name=scope + '1')

    broadcast_axes = []
    for axis_name in new_axis_names:
      if axis_name in axes_0:
        broadcast_axes.append(axes_0[axis_name])
      else:
        broadcast_axes.append(axes_1[axis_name])

    return labeled_tensor_0, labeled_tensor_1, Axes(broadcast_axes)


@tc.returns(types.FunctionType)
@tc.accepts(string_types, collections.Callable)
def define_unary_op(op_name, elementwise_function):
  """Define a unary operation for labeled tensors.

  Args:
    op_name: string name of the TensorFlow op.
    elementwise_function: function to call to evaluate the op on a single
      tf.Tensor object. This function must accept two arguments: a tf.Tensor
      object, and an optional `name`.

  Returns:
    Function defining the given op that acts on LabeledTensors.
  """

  default_name = 'lt_%s' % op_name

  @tc.returns(LabeledTensor)
  @tc.accepts(LabeledTensorLike, tc.Optional(string_types))
  def op(labeled_tensor, name=None):
    """LabeledTensor version of `tf.{op_name}`.

    See `tf.{op_name}` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.{op_name}` elementwise.
    """
    with ops.name_scope(name, default_name, [labeled_tensor]) as scope:
      labeled_tensor = convert_to_labeled_tensor(labeled_tensor)
      result_tensor = elementwise_function(labeled_tensor.tensor, name=scope)
      return LabeledTensor(result_tensor, labeled_tensor.axes)

  op.__doc__ = op.__doc__.format(op_name=op_name)
  op.__name__ = op_name

  return op


abs_function = define_unary_op('abs', math_ops.abs)
neg = define_unary_op('neg', math_ops.negative)
sign = define_unary_op('sign', math_ops.sign)
reciprocal = define_unary_op('reciprocal', math_ops.reciprocal)
square = define_unary_op('square', math_ops.square)
round_function = define_unary_op('round', math_ops.round)
sqrt = define_unary_op('sqrt', math_ops.sqrt)
rsqrt = define_unary_op('rsqrt', math_ops.rsqrt)
exp = define_unary_op('exp', math_ops.exp)
log = define_unary_op('log', math_ops.log)
ceil = define_unary_op('ceil', math_ops.ceil)
floor = define_unary_op('floor', math_ops.floor)
cos = define_unary_op('cos', math_ops.cos)
sin = define_unary_op('sin', math_ops.sin)
tan = define_unary_op('tan', math_ops.tan)
acos = define_unary_op('acos', math_ops.acos)
asin = define_unary_op('asin', math_ops.asin)
atan = define_unary_op('atan', math_ops.atan)
lgamma = define_unary_op('lgamma', math_ops.lgamma)
digamma = define_unary_op('digamma', math_ops.digamma)
erf = define_unary_op('erf', math_ops.erf)
erfc = define_unary_op('erfc', math_ops.erfc)
logical_not = define_unary_op('logical_not', math_ops.logical_not)
tanh = define_unary_op('tanh', math_ops.tanh)
sigmoid = define_unary_op('sigmoid', math_ops.sigmoid)


@tc.returns(types.FunctionType)
@tc.accepts(string_types, collections.Callable)
def define_binary_op(op_name, elementwise_function):
  """Define a binary operation that broadcasts labeled tensors.

  Args:
    op_name: string name of the TensorFlow op.
    elementwise_function: function to call to evaluate the op on tf.Tensor
      objects. This function must accept three arguments: two tf.Tensor objects,
      and an optional `name`.

  Returns:
    Function defining the given op that acts on LabeledTensors.
  """

  default_name = 'lt_%s' % op_name

  @tc.returns(LabeledTensor)
  @tc.accepts(LabeledTensorLike, LabeledTensorLike, tc.Optional(string_types))
  def op(labeled_tensor_0, labeled_tensor_1, name=None):
    """LabeledTensor version of `tf.{op_name}` with label based alignment.

    See `tf.{op_name}` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.{op_name}` elementwise.
    """
    with ops.name_scope(name, default_name,
                        [labeled_tensor_0, labeled_tensor_1]) as scope:

      align_0, align_1, broadcast_axes = align(labeled_tensor_0,
                                               labeled_tensor_1)

      tensor = elementwise_function(align_0.tensor, align_1.tensor, name=scope)

      return LabeledTensor(tensor, broadcast_axes)

  op.__doc__ = op.__doc__.format(op_name=op_name)
  op.__name__ = op_name

  return op


add = define_binary_op('add', math_ops.add)
sub = define_binary_op('sub', math_ops.subtract)
mul = define_binary_op('mul', math_ops.multiply)
div = define_binary_op('div', math_ops.div)
mod = define_binary_op('mod', math_ops.mod)
pow_function = define_binary_op('pow', math_ops.pow)

equal = define_binary_op('equal', math_ops.equal)
greater = define_binary_op('greater', math_ops.greater)
greater_equal = define_binary_op('greater_equal', math_ops.greater_equal)
not_equal = define_binary_op('not_equal', math_ops.not_equal)
less = define_binary_op('less', math_ops.less)
less_equal = define_binary_op('less_equal', math_ops.less_equal)
logical_and = define_binary_op('logical_and', math_ops.logical_and)
logical_or = define_binary_op('logical_or', math_ops.logical_or)
logical_xor = define_binary_op('logical_xor', math_ops.logical_xor)

maximum = define_binary_op('maximum', math_ops.maximum)
minimum = define_binary_op('minimum', math_ops.minimum)
squared_difference = define_binary_op('squared_difference',
                                      math_ops.squared_difference)
igamma = define_binary_op('igamma', math_ops.igamma)
igammac = define_binary_op('igammac', math_ops.igammac)
zeta = define_binary_op('zeta', math_ops.zeta)
polygamma = define_binary_op('polygamma', math_ops.polygamma)
