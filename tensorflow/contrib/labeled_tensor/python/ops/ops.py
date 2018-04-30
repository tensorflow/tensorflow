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
"""Non-core ops for LabeledTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import types

import numpy as np
from six import string_types

from tensorflow.contrib.labeled_tensor.python.ops import _typecheck as tc
from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import numerics
from tensorflow.python.ops import random_ops
from tensorflow.python.training import input  # pylint: disable=redefined-builtin


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensor, ops.Tensor, core.Axis,
            tc.Optional(string_types))
def _gather_1d_on_axis(labeled_tensor, indexer, axis, name=None):
  with ops.name_scope(name, 'lt_take', [labeled_tensor]) as scope:
    temp_axes = core.Axes([axis] + list(
        labeled_tensor.axes.remove(axis.name).values()))
    transposed = core.transpose(labeled_tensor, temp_axes.keys())
    indexed = core.LabeledTensor(
        array_ops.gather(transposed.tensor, indexer), temp_axes)
    return core.transpose(indexed, labeled_tensor.axes.keys(), name=scope)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Mapping(string_types,
                       tc.Union(slice, collections.Hashable, list)),
            tc.Optional(string_types))
def select(labeled_tensor, selection, name=None):
  """Slice out a subset of the tensor.

  Args:
    labeled_tensor: The input tensor.
    selection: A dictionary mapping an axis name to a scalar, slice or list of
      values to select. Currently supports two types of selections:
        (a) Any number of scalar and/or slice selections.
        (b) Exactly one list selection, without any scalars or slices.
    name: Optional op name.

  Returns:
    The selection as a `LabeledTensor`.

  Raises:
    ValueError: If the tensor doesn't have an axis in the selection or if
      that axis lacks labels.
    KeyError: If any labels in a selection are not found in the original axis.
    NotImplementedError: If you attempt to combine a list selection with
      scalar selection or another list selection.
  """
  with ops.name_scope(name, 'lt_select', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    slices = {}
    indexers = {}
    for axis_name, value in selection.items():
      if axis_name not in labeled_tensor.axes:
        raise ValueError(
            'The tensor does not have an axis named %s. Its axes are: %r' %
            (axis_name, labeled_tensor.axes.keys()))
      axis = labeled_tensor.axes[axis_name]
      if axis.labels is None:
        raise ValueError(
            'The axis named %s does not have labels. The axis is: %r' %
            (axis_name, axis))

      if isinstance(value, slice):
        # TODO(shoyer): consider deprecating using slices in favor of lists
        if value.start is None:
          start = None
        else:
          start = axis.index(value.start)

        if value.stop is None:
          stop = None
        else:
          # For now, follow the pandas convention of making labeled slices
          # inclusive of both bounds.
          stop = axis.index(value.stop) + 1

        if value.step is not None:
          raise NotImplementedError('slicing with a step is not yet supported')

        slices[axis_name] = slice(start, stop)

      # Needs to be after checking for slices, since slice objects claim to be
      # instances of collections.Hashable but hash() on them fails.
      elif isinstance(value, collections.Hashable):
        slices[axis_name] = axis.index(value)

      elif isinstance(value, list):
        if indexers:
          raise NotImplementedError(
              'select does not yet support more than one list selection at '
              'the same time')
        indexer = [axis.index(v) for v in value]
        indexers[axis_name] = ops.convert_to_tensor(indexer, dtype=dtypes.int64)

      else:
        # If type checking is working properly, this shouldn't be possible.
        raise TypeError('cannot handle arbitrary types')

    if indexers and slices:
      raise NotImplementedError(
          'select does not yet support combined scalar and list selection')

    # For now, handle array selection separately, because tf.gather_nd does
    # not support gradients yet. Later, using gather_nd will let us combine
    # these paths.
    if indexers:
      (axis_name, indexer), = indexers.items()
      axis = core.Axis(axis_name, selection[axis_name])
      return _gather_1d_on_axis(labeled_tensor, indexer, axis, name=scope)
    else:
      return core.slice_function(labeled_tensor, slices, name=scope)


@tc.returns(core.LabeledTensor)
@tc.accepts(
    tc.Collection(core.LabeledTensorLike), string_types,
    tc.Optional(string_types))
def concat(labeled_tensors, axis_name, name=None):
  """Concatenate tensors along a dimension.

  See tf.concat.

  Args:
    labeled_tensors: A list of input LabeledTensors.
    axis_name: The name of the axis along which to concatenate.
    name: Optional op name.

  Returns:
    The concatenated tensor.
    The coordinate labels for the concatenation dimension are also concatenated,
    if they are available for every tensor.

  Raises:
    ValueError: If fewer than one tensor inputs is provided, if the tensors
      have incompatible axes, or if `axis_name` isn't the name of an axis.
  """
  with ops.name_scope(name, 'lt_concat', labeled_tensors) as scope:
    labeled_tensors = [
        core.convert_to_labeled_tensor(lt) for lt in labeled_tensors
    ]

    if len(labeled_tensors) < 1:
      raise ValueError('concat expects at least 1 tensor, but received %s' %
                       labeled_tensors)

    # All tensors must have these axes.
    axes_0 = labeled_tensors[0].axes
    axis_names = list(axes_0.keys())

    if axis_name not in axis_names:
      raise ValueError('%s not in %s' % (axis_name, axis_names))

    shared_axes = axes_0.remove(axis_name)

    tensors = [labeled_tensors[0].tensor]
    concat_axis_list = [axes_0[axis_name]]
    for labeled_tensor in labeled_tensors[1:]:
      current_shared_axes = labeled_tensor.axes.remove(axis_name)
      if current_shared_axes != shared_axes:
        # TODO(shoyer): add more specific checks about what went wrong,
        # including raising AxisOrderError when appropriate
        raise ValueError('Mismatched shared axes: the first tensor '
                         'had axes %r but this tensor has axes %r.' %
                         (shared_axes, current_shared_axes))

      # Accumulate the axis labels, if they're available.
      concat_axis_list.append(labeled_tensor.axes[axis_name])
      tensors.append(labeled_tensor.tensor)

    concat_axis = core.concat_axes(concat_axis_list)
    concat_dimension = axis_names.index(axis_name)
    concat_tensor = array_ops.concat(tensors, concat_dimension, name=scope)
    values = list(axes_0.values())
    concat_axes = (values[:concat_dimension] + [concat_axis] +
                   values[concat_dimension + 1:])

    return core.LabeledTensor(concat_tensor, concat_axes)


# TODO(shoyer): rename pack/unpack to stack/unstack


@tc.returns(core.LabeledTensor)
@tc.accepts(
    tc.Collection(core.LabeledTensorLike),
    tc.Union(string_types, core.AxisLike), int, tc.Optional(string_types))
def pack(labeled_tensors, new_axis, axis_position=0, name=None):
  """Pack tensors along a new axis.

  See tf.pack.

  Args:
    labeled_tensors: The input tensors, which must have identical axes.
    new_axis: The name of the new axis, or a tuple containing the name
      and coordinate labels.
    axis_position: Optional integer position at which to insert the new axis.
    name: Optional op name.

  Returns:
    The packed tensors as a single LabeledTensor, with `new_axis` in the given
    `axis_position`.

  Raises:
    ValueError: If fewer than one input tensors is provided, or if the tensors
      don't have identical axes.
  """
  with ops.name_scope(name, 'lt_pack', labeled_tensors) as scope:
    labeled_tensors = [
        core.convert_to_labeled_tensor(lt) for lt in labeled_tensors
    ]

    if len(labeled_tensors) < 1:
      raise ValueError('pack expects at least 1 tensors, but received %s' %
                       labeled_tensors)

    axes_0 = labeled_tensors[0].axes
    for t in labeled_tensors:
      if t.axes != axes_0:
        raise ValueError('Non-identical axes. Expected %s but got %s' %
                         (axes_0, t.axes))

    pack_op = array_ops.stack(
        [t.tensor for t in labeled_tensors], axis=axis_position, name=scope)
    axes = list(axes_0.values())
    axes.insert(axis_position, new_axis)
    return core.LabeledTensor(pack_op, axes)


@tc.returns(tc.List(core.LabeledTensor))
@tc.accepts(core.LabeledTensorLike,
            tc.Optional(string_types), tc.Optional(string_types))
def unpack(labeled_tensor, axis_name=None, name=None):
  """Unpack the tensor.

  See tf.unpack.

  Args:
    labeled_tensor: The input tensor.
    axis_name: Optional name of axis to unpack. By default, the first axis is
      used.
    name: Optional op name.

  Returns:
    The list of unpacked LabeledTensors.

  Raises:
    ValueError: If `axis_name` is not an axis on the input.
  """
  with ops.name_scope(name, 'lt_unpack', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    axis_names = list(labeled_tensor.axes.keys())
    if axis_name is None:
      axis_name = axis_names[0]

    if axis_name not in axis_names:
      raise ValueError('%s not in %s' % (axis_name, axis_names))
    axis = axis_names.index(axis_name)

    unpack_ops = array_ops.unstack(labeled_tensor.tensor, axis=axis, name=scope)
    axes = [a for i, a in enumerate(labeled_tensor.axes.values()) if i != axis]
    return [core.LabeledTensor(t, axes) for t in unpack_ops]


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Collection(string_types),
            tc.Collection(tc.Union(string_types, core.AxisLike)),
            tc.Optional(string_types))
def reshape(labeled_tensor, existing_axes, new_axes, name=None):
  """Reshape specific axes of a LabeledTensor.

  Non-indicated axes remain in their original locations.

  Args:
    labeled_tensor: The input tensor.
    existing_axes: List of axis names found on the input tensor. These must
      appear sequentially in the list of axis names on the input. In other
      words, they must be a valid slice of `list(labeled_tensor.axes.keys())`.
    new_axes: List of strings, tuples of (axis_name, axis_value) or Axis objects
      providing new axes with which to replace `existing_axes` in the reshaped
      result. At most one element of `new_axes` may be a string, indicating an
      axis with unknown size.
    name: Optional op name.

  Returns:
    The reshaped LabeledTensor.

  Raises:
    ValueError: If `existing_axes` are not all axes on the input, or if more
     than one of `new_axes` has unknown size.
    AxisOrderError: If `existing_axes` are not a slice of axis names on the
      input.
  """
  with ops.name_scope(name, 'lt_reshape', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    original_axis_names = list(labeled_tensor.axes.keys())
    existing_axes = list(existing_axes)
    if not set(existing_axes) <= set(original_axis_names):
      raise ValueError('existing_axes %r are not contained in the set of axis '
                       'names %r on the input labeled tensor' %
                       (existing_axes, original_axis_names))

    start = original_axis_names.index(existing_axes[0])
    stop = original_axis_names.index(existing_axes[-1]) + 1

    if existing_axes != original_axis_names[start:stop]:
      # We could support existing_axes that aren't a slice by using transpose,
      # but that could lead to unpredictable performance consequences because
      # transposes are not free in TensorFlow. If we did transpose
      # automatically, the user might never realize that their data is being
      # produced with the wrong order. (The later will occur with some frequency
      # because of how broadcasting automatically choose axis order.)
      # So for now we've taken the strict approach.
      raise core.AxisOrderError(
          'existing_axes %r are not a slice of axis names %r on the input '
          'labeled tensor. Use `transpose` or `impose_axis_order` to reorder '
          'axes on the input explicitly.' %
          (existing_axes, original_axis_names))

    if sum(isinstance(axis, string_types) for axis in new_axes) > 1:
      raise ValueError(
          'at most one axis in new_axes can have unknown size. All other '
          'axes must have an indicated integer size or labels: %r' % new_axes)

    original_values = list(labeled_tensor.axes.values())
    axis_size = lambda axis: -1 if axis.size is None else axis.size
    shape = [axis_size(axis) for axis in original_values[:start]]
    for axis_ref in new_axes:
      if isinstance(axis_ref, string_types):
        shape.append(-1)
      else:
        axis = core.as_axis(axis_ref)
        shape.append(axis_size(axis))
    shape.extend(axis_size(axis) for axis in original_values[stop:])

    reshaped_tensor = array_ops.reshape(
        labeled_tensor.tensor, shape, name=scope)
    axes = original_values[:start] + list(new_axes) + original_values[stop:]
    return core.LabeledTensor(reshaped_tensor, axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike, string_types, string_types,
            tc.Optional(string_types))
def rename_axis(labeled_tensor, existing_name, new_name, name=None):
  """Rename an axis of LabeledTensor.

  Args:
    labeled_tensor: The input tensor.
    existing_name: Name for an existing axis on the input.
    new_name: Desired replacement name.
    name: Optional op name.

  Returns:
    LabeledTensor with renamed axis.

  Raises:
    ValueError: If `existing_name` is not an axis on the input.
  """
  with ops.name_scope(name, 'lt_rename_axis', [labeled_tensor]) as scope:
    if existing_name not in labeled_tensor.axes:
      raise ValueError('existing_name %r are not contained in the set of axis '
                       'names %r on the input labeled tensor' %
                       (existing_name, labeled_tensor.axes.keys()))
    new_axis = core.Axis(new_name, labeled_tensor.axes[existing_name].value)
    return reshape(labeled_tensor, [existing_name], [new_axis], name=scope)


@tc.returns(tc.List(core.LabeledTensor))
@tc.accepts(string_types, collections.Callable, int, bool,
            tc.Collection(core.LabeledTensorLike), bool,
            tc.Optional(string_types))
def _batch_helper(default_name,
                  batch_fn,
                  batch_size,
                  enqueue_many,
                  labeled_tensors,
                  allow_smaller_final_batch,
                  name=None):
  with ops.name_scope(name, default_name, labeled_tensors) as scope:
    labeled_tensors = [
        core.convert_to_labeled_tensor(lt) for lt in labeled_tensors
    ]

    batch_ops = batch_fn([t.tensor for t in labeled_tensors], scope)
    # TODO(shoyer): Remove this when they sanitize the TF API.
    if not isinstance(batch_ops, list):
      assert isinstance(batch_ops, ops.Tensor)
      batch_ops = [batch_ops]

    if allow_smaller_final_batch:
      batch_size = None

    @tc.returns(core.Axes)
    @tc.accepts(core.Axes)
    def output_axes(axes):
      if enqueue_many:
        if 'batch' not in axes or list(axes.keys()).index('batch') != 0:
          raise ValueError(
              'When enqueue_many is True, input tensors must have an axis '
              'called "batch" as their first dimension, '
              'but axes were %s' % axes)
        culled_axes = axes.remove('batch')
        return core.Axes([('batch', batch_size)] + list(culled_axes.values()))
      else:
        return core.Axes([('batch', batch_size)] + list(axes.values()))

    output_labeled_tensors = []
    for i, tensor in enumerate(batch_ops):
      axes = output_axes(labeled_tensors[i].axes)
      output_labeled_tensors.append(core.LabeledTensor(tensor, axes))

    return output_labeled_tensors


@tc.returns(tc.List(core.LabeledTensor))
@tc.accepts(
    tc.Collection(core.LabeledTensorLike), int, int, int, bool, bool,
    tc.Optional(string_types))
def batch(labeled_tensors,
          batch_size,
          num_threads=1,
          capacity=32,
          enqueue_many=False,
          allow_smaller_final_batch=False,
          name=None):
  """Rebatch a tensor.

  See tf.batch.

  Args:
    labeled_tensors: The input tensors.
    batch_size: The output batch size.
    num_threads: See tf.batch.
    capacity: See tf.batch.
    enqueue_many: If true, the input tensors must contain a 'batch' axis as
      their first axis.
      If false, the input tensors must not contain a 'batch' axis.
      See tf.batch.
    allow_smaller_final_batch: See tf.batch.
    name: Optional op name.

  Returns:
    The rebatched tensors.
    If enqueue_many is false, the output tensors will have a new 'batch' axis
      as their first axis.

  Raises:
    ValueError: If enqueue_many is True and the first axis of the tensors
      isn't "batch".
  """

  def fn(tensors, scope):
    return input.batch(
        tensors,
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=capacity,
        enqueue_many=enqueue_many,
        allow_smaller_final_batch=allow_smaller_final_batch,
        name=scope)

  return _batch_helper('lt_batch', fn, batch_size, enqueue_many,
                       labeled_tensors, allow_smaller_final_batch, name)


@tc.returns(tc.List(core.LabeledTensor))
@tc.accepts(
    tc.Collection(core.LabeledTensorLike), int, int, int, bool, int,
    tc.Optional(int), bool, tc.Optional(string_types))
def shuffle_batch(labeled_tensors,
                  batch_size,
                  num_threads=1,
                  capacity=32,
                  enqueue_many=False,
                  min_after_dequeue=0,
                  seed=None,
                  allow_smaller_final_batch=False,
                  name=None):
  """Rebatch a tensor, with shuffling.

  See tf.batch.

  Args:
    labeled_tensors: The input tensors.
    batch_size: The output batch size.
    num_threads: See tf.batch.
    capacity: See tf.batch.
    enqueue_many: If true, the input tensors must contain a 'batch' axis as
      their first axis.
      If false, the input tensors must not contain a 'batch' axis.
      See tf.batch.
    min_after_dequeue: Minimum number of elements in the queue after a dequeue,
      used to ensure mixing.
    seed: Optional random seed.
    allow_smaller_final_batch: See tf.batch.
    name: Optional op name.

  Returns:
    The rebatched tensors.
    If enqueue_many is false, the output tensors will have a new 'batch' axis
      as their first axis.

  Raises:
    ValueError: If enqueue_many is True and the first axis of the tensors
      isn't "batch".
  """

  def fn(tensors, scope):
    return input.shuffle_batch(
        tensors,
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=capacity,
        enqueue_many=enqueue_many,
        min_after_dequeue=min_after_dequeue,
        seed=seed,
        allow_smaller_final_batch=allow_smaller_final_batch,
        name=scope)

  return _batch_helper('lt_shuffle_batch', fn, batch_size, enqueue_many,
                       labeled_tensors, allow_smaller_final_batch, name)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Mapping(string_types, int),
            tc.Optional(int), tc.Optional(string_types))
def random_crop(labeled_tensor, shape_map, seed=None, name=None):
  """Randomly crops a tensor to a given size.

  See tf.random_crop.

  Args:
    labeled_tensor: The input tensor.
    shape_map: A dictionary mapping axis names to the size of the random crop
      for that dimension.
    seed: An optional random seed.
    name: An optional op name.

  Returns:
    A tensor of the same rank as `labeled_tensor`, cropped randomly in the
    selected dimensions.

  Raises:
    ValueError: If the shape map contains an axis name not in the input tensor.
  """
  with ops.name_scope(name, 'lt_random_crop', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    for axis_name in shape_map:
      if axis_name not in labeled_tensor.axes:
        raise ValueError('Selection axis %s not in axes %s' %
                         (axis_name, labeled_tensor.axes))

    shape = []
    axes = []
    for axis in labeled_tensor.axes.values():
      if axis.name in shape_map:
        size = shape_map[axis.name]
        shape.append(size)
        # We lose labels for the axes we crop, leaving just the size.
        axes.append((axis.name, size))
      else:
        shape.append(len(axis))
        axes.append(axis)

    crop_op = random_ops.random_crop(
        labeled_tensor.tensor, shape, seed=seed, name=scope)

    return core.LabeledTensor(crop_op, axes)


# TODO(shoyer): Allow the user to select the axis over which to map.
@tc.returns(core.LabeledTensor)
@tc.accepts(collections.Callable, core.LabeledTensorLike,
            tc.Optional(string_types))
def map_fn(fn, labeled_tensor, name=None):
  """Map on the list of tensors unpacked from labeled_tensor.

  See tf.map_fn.

  Args:
    fn: The function to apply to each unpacked LabeledTensor.
      It should have type LabeledTensor -> LabeledTensor.
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    A tensor that packs the results of applying fn to the list of tensors
    unpacked from labeled_tensor.
  """
  with ops.name_scope(name, 'lt_map_fn', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    unpack_lts = unpack(labeled_tensor)

    # TODO(ericmc): Fix this upstream.
    if labeled_tensor.dtype == dtypes.string:
      # We must construct the full graph here, because functional_ops.map_fn
      # doesn't work for string-valued tensors.
      # Constructing the full graph may be slow.
      map_lts = [fn(t) for t in unpack_lts]
      return pack(map_lts, list(labeled_tensor.axes.values())[0], name=scope)
    else:
      # Figure out what the axis labels should be, but use tf.map_fn to
      # construct the graph because it's efficient.
      # It may be slow to construct the full graph, so we infer the labels from
      # the first element.
      # TODO(ericmc): This builds a subgraph which then gets thrown away.
      # Find a more elegant solution.
      first_map_lt = fn(unpack_lts[0])
      final_axes = list(labeled_tensor.axes.values())[:1] + list(
          first_map_lt.axes.values())

      @tc.returns(ops.Tensor)
      @tc.accepts(ops.Tensor)
      def tf_fn(tensor):
        original_axes = list(labeled_tensor.axes.values())[1:]
        tensor_lt = core.LabeledTensor(tensor, original_axes)
        return fn(tensor_lt).tensor

      map_op = functional_ops.map_fn(tf_fn, labeled_tensor.tensor)
      map_lt = core.LabeledTensor(map_op, final_axes)

      return core.identity(map_lt, name=scope)


@tc.returns(core.LabeledTensor)
@tc.accepts(collections.Callable, core.LabeledTensorLike,
            core.LabeledTensorLike, tc.Optional(string_types))
def foldl(fn, labeled_tensor, initial_value, name=None):
  """Left fold on the list of tensors unpacked from labeled_tensor.

  See tf.foldl.

  Args:
    fn: The function to apply to each unpacked LabeledTensor.
      It should have type (LabeledTensor, LabeledTensor) -> LabeledTensor.
      Its arguments are (accumulated_value, next_value).
    labeled_tensor: The input tensor.
    initial_value: The initial value of the accumulator.
    name: Optional op name.

  Returns:
    The accumulated value.
  """
  with ops.name_scope(name, 'lt_foldl',
                      [labeled_tensor, initial_value]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)
    initial_value = core.convert_to_labeled_tensor(initial_value)

    @tc.returns(ops.Tensor)
    @tc.accepts(ops.Tensor, ops.Tensor)
    def tf_fn(accumulator, next_element):
      accumulator_lt = core.LabeledTensor(accumulator, initial_value.axes)
      next_element_lt = core.LabeledTensor(
          next_element, list(labeled_tensor.axes.values())[1:])
      return fn(accumulator_lt, next_element_lt).tensor

    foldl_op = functional_ops.foldl(
        tf_fn, labeled_tensor.tensor, initializer=initial_value.tensor)
    foldl_lt = core.LabeledTensor(foldl_op, initial_value.axes)

    return core.identity(foldl_lt, name=scope)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Optional(tc.Collection(string_types)), tc.Optional(string_types))
def squeeze(labeled_tensor, axis_names=None, name=None):
  """Remove size-1 dimensions.

  See tf.squeeze.

  Args:
    labeled_tensor: The input tensor.
    axis_names: The names of the dimensions to remove, or None to remove
      all size-1 dimensions.
    name: Optional op name.

  Returns:
    A tensor with the specified dimensions removed.

  Raises:
    ValueError: If the named axes are not in the tensor, or if they are
      not size-1.
  """
  with ops.name_scope(name, 'lt_squeeze', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    if axis_names is None:
      axis_names = [a.name for a in labeled_tensor.axes.values() if len(a) == 1]

    for axis_name in axis_names:
      if axis_name not in labeled_tensor.axes:
        raise ValueError('axis %s is not in tensor axes %s' %
                         (axis_name, labeled_tensor.axes))
      elif len(labeled_tensor.axes[axis_name]) != 1:
        raise ValueError(
            'cannot squeeze axis with size greater than 1: (%s, %s)' %
            (axis_name, labeled_tensor.axes[axis_name]))

    squeeze_dimensions = []
    axes = []
    for i, axis in enumerate(labeled_tensor.axes.values()):
      if axis.name in axis_names:
        squeeze_dimensions.append(i)
      else:
        axes.append(axis)

    if squeeze_dimensions:
      squeeze_op = array_ops.squeeze(
          labeled_tensor.tensor, squeeze_dimensions, name=scope)
    else:
      squeeze_op = array_ops.identity(labeled_tensor.tensor, name=scope)

    return core.LabeledTensor(squeeze_op, axes)


# pylint: disable=invalid-name
ReduceAxis = tc.Union(string_types,
                      tc.Tuple(string_types, collections.Hashable))
ReduceAxes = tc.Optional(tc.Union(ReduceAxis, tc.Collection(ReduceAxis)))
# pylint: enable=invalid-name


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike, core.LabeledTensorLike,
            tc.Optional(string_types))
def matmul(a, b, name=None):
  """Matrix multiply two tensors with rank 1 or 2.

  If both tensors have rank 2, a matrix-matrix product is performed.
  If one tensor has rank 1 and the other has rank 2, then a matrix-vector
  product is performed.
  If both tensors have rank 1, then a vector dot-product is performed.
  (This behavior matches that of `numpy.dot`.)

  Both tensors must share exactly one dimension in common, which is the
  dimension the operation is summed along. The inputs will be automatically
  transposed if necessary as part of the matmul op.

  We intend to eventually support `matmul` on higher rank input, and also
  eventually support summing over any number shared dimensions (via an `axis`
  argument), but neither of these features has been implemented yet.

  Args:
    a: First LabeledTensor.
    b: Second LabeledTensor.
    name: Optional op name.

  Returns:
    LabeledTensor with the result of matrix multiplication. Axes are ordered by
    the current axis_order_scope, if set, or in or order of appearance on the
    inputs.

  Raises:
    NotImplementedError: If inputs have rank >2 or share multiple axes.
    ValueError: If the inputs have rank 0 or do not share any axes.
  """
  with ops.name_scope(name, 'lt_matmul', [a, b]) as scope:

    a = core.convert_to_labeled_tensor(a)
    b = core.convert_to_labeled_tensor(b)

    if len(a.axes) > 2 or len(b.axes) > 2:
      # We could pass batched inputs to tf.matmul to make this work, but we
      # would also need to use tf.tile and/or tf.transpose. These are more
      # expensive than doing reshapes, so it's not clear if it's a good idea to
      # do this automatically.
      raise NotImplementedError(
          'matmul currently requires inputs with rank 2 or less, but '
          'inputs have ranks %r and %r' % (len(a.axes), len(b.axes)))

    if not a.axes or not b.axes:
      raise ValueError(
          'matmul currently requires inputs with at least rank 1, but '
          'inputs have ranks %r and %r' % (len(a.axes), len(b.axes)))

    shared_axes = set(a.axes) & set(b.axes)
    if len(shared_axes) > 1:
      raise NotImplementedError(
          'matmul does not yet support summing over multiple shared axes: %r. '
          'Use transpose and reshape to create a single shared axis to sum '
          'over.' % shared_axes)
    if not shared_axes:
      raise ValueError('there must have exactly one axis in common between '
                       'input to matmul: %r, %r' %
                       (a.axes.keys(), b.axes.keys()))
    shared_axis, = shared_axes

    if a.axes[shared_axis] != b.axes[shared_axis]:
      raise ValueError('axis %r does not match on input arguments: %r vs %r' %
                       (shared_axis, a.axes[shared_axis].value,
                        b.axes[shared_axis].value))

    result_axes = []
    for axes in [a.axes, b.axes]:
      for axis in axes.values():
        if axis.name != shared_axis:
          result_axes.append(axis)

    axis_scope_order = core.get_axis_order()
    if axis_scope_order is not None:
      result_axis_names = [axis.name for axis in result_axes]
      new_axis_names = [
          name for name in axis_scope_order if name in result_axis_names
      ]
      if new_axis_names != result_axis_names:
        # switch a and b
        b, a = a, b
        # result_axes is a list of length 1 or 2
        result_axes = result_axes[::-1]

    squeeze_dims = []

    if len(a.axes) == 1:
      a_tensor = array_ops.reshape(a.tensor, (1, -1))
      squeeze_dims.append(0)
      transpose_a = False
    else:
      a_tensor = a.tensor
      transpose_a = list(a.axes.keys()).index(shared_axis) == 0

    if len(b.axes) == 1:
      b_tensor = array_ops.reshape(b.tensor, (-1, 1))
      squeeze_dims.append(1)
      transpose_b = False
    else:
      b_tensor = b.tensor
      transpose_b = list(b.axes.keys()).index(shared_axis) == 1

    result_op = math_ops.matmul(
        a_tensor, b_tensor, transpose_a=transpose_a, transpose_b=transpose_b)

    if squeeze_dims:
      result_op = array_ops.squeeze(result_op, squeeze_dims)
    result_op = array_ops.identity(result_op, name=scope)

    return core.LabeledTensor(result_op, result_axes)


@tc.returns(types.FunctionType)
@tc.accepts(string_types, collections.Callable)
def define_reduce_op(op_name, reduce_fn):
  """Define a reduction op for labeled tensors.

  Args:
    op_name: string name of the TensorFlow op.
    reduce_fn: function to call to evaluate the op on a tf.Tensor.

  Returns:
    Function defining the given reduction op that acts on a LabeledTensor.
  """

  default_name = 'lt_%s' % op_name

  @tc.returns(core.LabeledTensor)
  @tc.accepts(core.LabeledTensorLike, ReduceAxes, tc.Optional(string_types))
  def op(labeled_tensor, axes=None, name=None):
    """Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.{op_name}` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    """
    with ops.name_scope(name, default_name, [labeled_tensor]) as scope:
      labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

      if axes is None:
        axes = labeled_tensor.axes.keys()

      if isinstance(axes, (string_types, tuple)):
        axes = [axes]

      reduction_axes = {}
      axes_to_squeeze = []
      for a in axes:
        if isinstance(a, string_types):
          # We squeeze out this axis.
          reduction_axes[a] = a
          axes_to_squeeze.append(a)
        else:
          # We keep this axis, with the user-provided labels.
          (axis_name, label) = a
          if label is not None:
            # The input was a single label, so make it a list so it can be
            # turned into an Axis.
            label = [label]
          reduction_axes[axis_name] = (axis_name, label)

      for axis_name in reduction_axes:
        if axis_name not in labeled_tensor.axes:
          raise ValueError('Axis %s not in axes %s' %
                           (axis_name, labeled_tensor.axes))

      intermediate_axes = []
      reduction_dimensions = []
      for i, axis in enumerate(labeled_tensor.axes.values()):
        if axis.name in reduction_axes:
          intermediate_axes.append(reduction_axes[axis.name])
          reduction_dimensions.append(i)
        else:
          intermediate_axes.append(axis)

      reduce_op = reduce_fn(
          labeled_tensor.tensor, reduction_dimensions, keepdims=True)
      reduce_lt = core.LabeledTensor(reduce_op, intermediate_axes)

      return squeeze(reduce_lt, axes_to_squeeze, name=scope)

  op.__doc__ = op.__doc__.format(op_name=op_name)
  op.__name__ = op_name

  return op


reduce_all = define_reduce_op('reduce_all', math_ops.reduce_all)
reduce_any = define_reduce_op('reduce_any', math_ops.reduce_any)
reduce_logsumexp = define_reduce_op('reduce_logsumexp',
                                    math_ops.reduce_logsumexp)
reduce_max = define_reduce_op('reduce_max', math_ops.reduce_max)
reduce_mean = define_reduce_op('reduce_mean', math_ops.reduce_mean)
reduce_min = define_reduce_op('reduce_min', math_ops.reduce_min)
reduce_prod = define_reduce_op('reduce_prod', math_ops.reduce_prod)
reduce_sum = define_reduce_op('reduce_sum', math_ops.reduce_sum)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Mapping(str, tc.Union(int, ops.Tensor)),
            tc.Optional(string_types))
def tile(labeled_tensor, multiples, name=None):
  """Constructs a tensor by tiling a given tensor.

  Only axes without tick-labels can be tiled. (Otherwise, axis labels on tiled
  tensors would no longer be unique.)

  See lt.tile.

  Args:
    labeled_tensor: The input tensor.
    multiples: A mapping where the keys are axis names and the values are the
      integer number of times to tile along that axis. Only axes with a multiple
      different than 1 need be included.
    name: Optional op name.

  Returns:
    A tensor with the indicated axes tiled.

  Raises:
    ValueError: If the tiled axes are not axes in the input tensor, or if any
      axes in multiples have tick labels.
  """
  with ops.name_scope(name, 'lt_tile', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    if not set(multiples.keys()) <= set(labeled_tensor.axes.keys()):
      raise ValueError('tile axes %r are not contained in the set of axis '
                       'names %r on the input labeled tensor' %
                       (multiples.keys(), labeled_tensor.axes))

    labeled_axes = [
        name for name in multiples
        if labeled_tensor.axes[name].labels is not None
    ]
    if labeled_axes:
      raise ValueError('cannot tile axes with tick labels: %r' % labeled_axes)

    multiples_list = [multiples.get(name, 1) for name in labeled_tensor.axes]
    tile_op = array_ops.tile(labeled_tensor.tensor, multiples_list, name=scope)

    new_axes = [
        axis.name if axis.labels is None else axis
        for axis in labeled_tensor.axes.values()
    ]
    return core.LabeledTensor(tile_op, new_axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Mapping(str, tc.Tuple(core.AxisValue, core.AxisValue)),
            string_types, tc.Optional(string_types))
def pad(labeled_tensor, paddings, mode='CONSTANT', name=None):
  """Pads a tensor.

  See tf.pad.

  Args:
    labeled_tensor: The input tensor.
    paddings: A mapping where the keys are axis names and the values are
      tuples where the first element is the padding to insert at the beginning
      of the axis and the second is the padding to insert at the end of the
      axis.
    mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
    name: Optional op name.

  Returns:
    A tensor with the indicated axes padded, optionally with those axes extended
    with the provided labels.

  Raises:
    ValueError: If the padded axes are not axes in the input tensor.
  """
  with ops.name_scope(name, 'lt_pad', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

    if not set(paddings.keys()) <= set(labeled_tensor.axes.keys()):
      raise ValueError('pad axes %r are not contained in the set of axis '
                       'names %r on the input labeled tensor' %
                       (paddings.keys(), labeled_tensor.axes))

    new_axes = []
    padding_pairs = []
    for name, axis in labeled_tensor.axes.items():
      if name in paddings:
        padding_before, padding_after = paddings[name]
        axis_before = core.Axis(name, padding_before)
        axis_after = core.Axis(name, padding_after)
        new_axes.append(core.concat_axes([axis_before, axis, axis_after]))
        padding_pairs.append((len(axis_before), len(axis_after)))
      else:
        new_axes.append(axis)
        padding_pairs.append((0, 0))

    pad_op = array_ops.pad(labeled_tensor.tensor,
                           padding_pairs,
                           mode,
                           name=scope)

    return core.LabeledTensor(pad_op, new_axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(
    tc.Union(np.ndarray, list, tuple, core.Scalar),
    tc.Optional(dtypes.DType),
    tc.Optional(
        tc.Union(core.Axes, tc.Collection(
            tc.Union(string_types, core.AxisLike)))), tc.Optional(string_types))
def constant(value, dtype=None, axes=None, name=None):
  """Creates a constant tensor.

  If `axes` includes any strings, shape is inferred from `value`. Otherwise,
  the sizes of the given `axes` are used to set `shape` for `tf.constant`.

  See tf.constant for more details.

  Args:
    value: The input tensor.
    dtype: The type of the returned tensor.
    axes: Optional Axes, list of strings or list of objects coercible to Axis
      objects. By default, axes are assumed to be an empty list (i.e., `value`
      is treated as a scalar).
    name: Optional op name.

  Returns:
    The tensor with elements set to zero.
  """
  with ops.name_scope(name, 'lt_constant', [value]) as scope:

    if axes is None:
      axes = []

    if isinstance(axes, core.Axes):
      axes = axes.values()

    if any(isinstance(ax, string_types) for ax in axes):
      # need to infer shape
      shape = None
    else:
      # axes already indicate shape
      axes = [core.as_axis(a) for a in axes]
      shape = [a.size for a in axes]

    op = array_ops.constant(value, dtype=dtype, shape=shape, name=scope)
    return core.LabeledTensor(op, axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Optional(dtypes.DType), tc.Optional(string_types))
def zeros_like(labeled_tensor, dtype=None, name=None):
  """Creates an identical tensor with all elements set to zero.

  Args:
    labeled_tensor: The input tensor.
    dtype: The type of the returned tensor.
    name: Optional op name.

  Returns:
    The tensor with elements set to zero.
  """
  with ops.name_scope(name, 'lt_zeros_like', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)
    op = array_ops.zeros_like(labeled_tensor.tensor, dtype=dtype, name=scope)
    return core.LabeledTensor(op, labeled_tensor.axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Optional(dtypes.DType), tc.Optional(string_types))
def ones_like(labeled_tensor, dtype=None, name=None):
  """Creates an identical tensor with all elements set to one.

  Args:
    labeled_tensor: The input tensor.
    dtype: The type of the returned tensor.
    name: Optional op name.

  Returns:
    The tensor with elements set to one.
  """
  with ops.name_scope(name, 'lt_ones_like', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)
    op = array_ops.ones_like(labeled_tensor.tensor, dtype=dtype, name=scope)
    return core.LabeledTensor(op, labeled_tensor.axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike,
            tc.Optional(dtypes.DType), tc.Optional(string_types))
def cast(labeled_tensor, dtype=None, name=None):
  """Casts a labeled tensor to a new type.

  Args:
    labeled_tensor: The input tensor.
    dtype: The type of the returned tensor.
    name: Optional op name.

  Returns:
    A labeled tensor with the new dtype.
  """
  with ops.name_scope(name, 'lt_cast', [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)
    op = math_ops.cast(labeled_tensor.tensor, dtype=dtype, name=scope)
    return core.LabeledTensor(op, labeled_tensor.axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike, string_types, tc.Optional(string_types))
def verify_tensor_all_finite(labeled_tensor, message, name=None):
  """Asserts a tensor doesn't contain NaNs or Infs.

  See tf.verify_tensor_all_finite.

  Args:
    labeled_tensor: The input tensor.
    message: Message to log on failure.
    name: Optional op name.

  Returns:
    The input tensor.
  """
  with ops.name_scope(name, 'lt_verify_tensor_all_finite',
                      [labeled_tensor]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)
    op = numerics.verify_tensor_all_finite(
        labeled_tensor.tensor, msg=message, name=scope)
    return core.LabeledTensor(op, labeled_tensor.axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike, core.LabeledTensorLike,
            tc.Optional(string_types))
def boolean_mask(labeled_tensor, mask, name=None):
  """Apply a boolean mask to a labeled tensor.

  Unlike `tf.boolean_mask`, this currently only works on 1-dimensional masks.
  The mask is applied to the first axis of `labeled_tensor`. Labels on the first
  axis are removed, because True indices in `mask` may not be known dynamically.

  Args:
    labeled_tensor: The input tensor.
    mask: The type of the returned tensor.
    name: Optional op name.

  Returns:
    The masked labeled tensor.

  Raises:
    ValueError: if the first axis of the mask
  """
  with ops.name_scope(name, 'lt_boolean_mask', [labeled_tensor, mask]) as scope:
    labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)
    mask = core.convert_to_labeled_tensor(mask)

    if len(mask.axes) > 1:
      raise NotImplementedError(
          "LabeledTensor's boolean_mask currently only supports 1D masks")
    mask_axis = list(mask.axes.values())[0]
    lt_axis = list(labeled_tensor.axes.values())[0]
    if mask_axis != lt_axis:
      raise ValueError('the first axis of the labeled tensor and the mask '
                       'are not equal:\n%r\n%r' % (lt_axis, mask_axis))
    op = array_ops.boolean_mask(labeled_tensor.tensor, mask.tensor, name=scope)
    # TODO(shoyer): attempt to infer labels for the masked values, by calling
    # tf.contrib.util.constant_value on the mask?
    axes = [lt_axis.name] + list(labeled_tensor.axes.values())[1:]
    return core.LabeledTensor(op, axes)


@tc.returns(core.LabeledTensor)
@tc.accepts(core.LabeledTensorLike, core.LabeledTensorLike,
            core.LabeledTensorLike, tc.Optional(string_types))
def where(condition, x, y, name=None):
  """Return elements from x or y depending on condition.

  See `tf.where` for more details. This function currently only implements the
  three argument version of where.

  Args:
    condition: LabeledTensor of type `bool`.
    x: LabeledTensor for values where condition is true.
    y: LabeledTensor for values where condition is false.
    name: Optional op name.

  Returns:
    The labeled tensor with values according to condition.

  Raises:
    ValueError: if `x` and `y` have different axes, or if the axes of `x` do not
      start with the axes of `condition`.
  """
  with ops.name_scope(name, 'lt_where', [condition, x, y]) as scope:
    condition = core.convert_to_labeled_tensor(condition)
    x = core.convert_to_labeled_tensor(x)
    y = core.convert_to_labeled_tensor(y)

    if not condition.axes == x.axes == y.axes:
      raise ValueError('all inputs to `where` must have equal axes')

    op = array_ops.where(condition.tensor, x.tensor, y.tensor, name=scope)
    return core.LabeledTensor(op, x.axes)
