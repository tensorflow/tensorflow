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
"""Operators specific to data structures: list append, subscripts, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops


# TODO(mdan): Once control flow supports objects, repackage as a class.


def new_list(iterable=None):
  """The list constructor.

  Args:
    iterable: Optional elements to fill the list with.

  Returns:
    A list-like object. The exact return value depends on the initial elements.
  """
  if iterable:
    elements = tuple(iterable)
  else:
    elements = ()

  if elements:
    # When the list contains elements, it is assumed to be a "Python" lvalue
    # list.
    return _py_list_new(elements)
  return tf_tensor_list_new(elements)


def tf_tensor_array_new(elements, element_dtype=None, element_shape=None):
  """Overload of new_list that stages a Tensor list creation."""
  elements = tuple(ops.convert_to_tensor(el) for el in elements)

  all_dtypes = set(el.dtype for el in elements)
  if len(all_dtypes) == 1:
    inferred_dtype, = tuple(all_dtypes)
    if element_dtype is not None and element_dtype != inferred_dtype:
      raise ValueError(
          'incompatible dtype; specified: {}, inferred from {}: {}'.format(
              element_dtype, elements, inferred_dtype))
  elif len(all_dtypes) > 1:
    raise ValueError(
        'TensorArray requires all elements to have the same dtype:'
        ' {}'.format(elements))
  else:
    if element_dtype is None:
      raise ValueError('dtype is required to create an empty TensorArray')

  all_shapes = set(tuple(el.shape.as_list()) for el in elements)
  if len(all_shapes) == 1:
    inferred_shape, = tuple(all_shapes)
    if element_shape is not None and element_shape != inferred_shape:
      raise ValueError(
          'incompatible shape; specified: {}, inferred from {}: {}'.format(
              element_shape, elements, inferred_shape))
  elif len(all_shapes) > 1:
    raise ValueError(
        'TensorArray requires all elements to have the same shape:'
        ' {}'.format(elements))
    # TODO(mdan): We may want to allow different shapes with infer_shape=False.
  else:
    inferred_shape = None

  if element_dtype is None:
    element_dtype = inferred_dtype
  if element_shape is None:
    element_shape = inferred_shape

  l = tensor_array_ops.TensorArray(
      dtype=element_dtype,
      size=len(elements),
      dynamic_size=True,
      infer_shape=(element_shape is None),
      element_shape=element_shape)
  for i, el in enumerate(elements):
    l = l.write(i, el)
  return l


def tf_tensor_list_new(elements, element_dtype=None, element_shape=None):
  """Overload of new_list that stages a Tensor list creation."""
  if tensor_util.is_tf_type(elements):
    if element_shape is not None:
      raise ValueError(
          'element shape may not be specified when creating list from tensor')
    element_shape = array_ops.shape(elements)[1:]
    l = list_ops.tensor_list_from_tensor(elements, element_shape=element_shape)
    return l

  elements = tuple(ops.convert_to_tensor(el) for el in elements)

  all_dtypes = set(el.dtype for el in elements)
  if len(all_dtypes) == 1:
    inferred_dtype = tuple(all_dtypes)[0]
    if element_dtype is not None and element_dtype != inferred_dtype:
      raise ValueError(
          'incompatible dtype; specified: {}, inferred from {}: {}'.format(
              element_dtype, elements, inferred_dtype))
  elif all_dtypes:
    # Heterogeneous lists are ok.
    if element_dtype is not None:
      raise ValueError(
          'specified dtype {} is inconsistent with that of elements {}'.format(
              element_dtype, elements))
    inferred_dtype = dtypes.variant
  else:
    inferred_dtype = dtypes.variant

  all_shapes = set(tuple(el.shape.as_list()) for el in elements)
  if len(all_shapes) == 1:
    inferred_shape = array_ops.shape(elements[0])
    if element_shape is not None and element_shape != inferred_shape:
      raise ValueError(
          'incompatible shape; specified: {}, inferred from {}: {}'.format(
              element_shape, elements, inferred_shape))
  elif all_shapes:
    # Heterogeneous lists are ok.
    if element_shape is not None:
      raise ValueError(
          'specified shape {} is inconsistent with that of elements {}'.format(
              element_shape, elements))
    inferred_shape = constant_op.constant(-1)  # unknown shape, by convention
  else:
    inferred_shape = constant_op.constant(-1)  # unknown shape, by convention

  if element_dtype is None:
    element_dtype = inferred_dtype
  if element_shape is None:
    element_shape = inferred_shape

  element_shape = ops.convert_to_tensor(element_shape, dtype=dtypes.int32)
  l = list_ops.empty_tensor_list(
      element_shape=element_shape, element_dtype=element_dtype)
  for el in elements:
    l = list_ops.tensor_list_push_back(l, el)
  return l


def _py_list_new(elements):
  """Overload of new_list that creates a Python list."""
  return list(elements)


def list_append(list_, x):
  """The list append function.

  Note: it is unspecified where list_ will be mutated or not. If list_ is
  a TensorFlow entity, it will not be typically mutated. If list_ is a plain
  list, it will be. In general, if the list is mutated then the return value
  should point to the original entity.

  Args:
    list_: An entity that supports append semantics.
    x: The element to append.

  Returns:
    Same as list_, after the append was performed.

  Raises:
    ValueError: if list_ is not of a known list-like type.
  """
  if isinstance(list_, tensor_array_ops.TensorArray):
    return _tf_tensorarray_append(list_, x)
  elif tensor_util.is_tf_type(list_):
    if list_.dtype == dtypes.variant:
      return _tf_tensor_list_append(list_, x)
    else:
      raise ValueError(
          'tensor lists are expected to be Tensors with dtype=tf.variant,'
          ' instead found %s' % list_)
  else:
    return _py_list_append(list_, x)


def _tf_tensor_list_append(list_, x):
  """Overload of list_append that stages a Tensor list write."""
  def empty_list_of_elements_like_x():
    tensor_x = ops.convert_to_tensor(x)
    return list_ops.empty_tensor_list(
        element_shape=array_ops.shape(tensor_x),
        element_dtype=tensor_x.dtype)

  list_ = control_flow_ops.cond(
      list_ops.tensor_list_length(list_) > 0,
      lambda: list_,
      empty_list_of_elements_like_x,
  )
  return list_ops.tensor_list_push_back(list_, x)


def _tf_tensorarray_append(list_, x):
  """Overload of list_append that stages a TensorArray write."""
  return list_.write(list_.size(), x)


def _py_list_append(list_, x):
  """Overload of list_append that executes a Python list append."""
  # Revert to the original call.
  list_.append(x)
  return list_


class ListPopOpts(
    collections.namedtuple('ListPopOpts', ('element_dtype', 'element_shape'))):
  pass


def list_pop(list_, i, opts):
  """The list pop function.

  Note: it is unspecified where list_ will be mutated or not. If list_ is
  a TensorFlow entity, it will not be typically mutated. If list_ is a plain
  list, it will be. In general, if the list is mutated then the return value
  should point to the original entity.

  Args:
    list_: An entity that supports pop semantics.
    i: Optional index to pop from. May be None.
    opts: A ListPopOpts.

  Returns:
    Tuple (x, out_list_):
      out_list_: same as list_, after the removal was performed.
      x: the removed element value.

  Raises:
    ValueError: if list_ is not of a known list-like type or the operation is
    not supported for that type.
  """
  assert isinstance(opts, ListPopOpts)

  if isinstance(list_, tensor_array_ops.TensorArray):
    raise ValueError('TensorArray does not support item removal')
  elif tensor_util.is_tf_type(list_):
    if list_.dtype == dtypes.variant:
      return _tf_tensor_list_pop(list_, i, opts)
    else:
      raise ValueError(
          'tensor lists are expected to be Tensors with dtype=tf.variant,'
          ' instead found %s' % list_)
  else:
    return _py_list_pop(list_, i)


def _tf_tensor_list_pop(list_, i, opts):
  """Overload of list_pop that stages a Tensor list pop."""
  if i is not None:
    raise NotImplementedError('tensor lists only support removing from the end')

  if opts.element_dtype is None:
    raise ValueError('cannot pop from a list without knowing its element '
                     'type; use set_element_type to annotate it')
  if opts.element_shape is None:
    raise ValueError('cannot pop from a list without knowing its element '
                     'shape; use set_element_type to annotate it')
  list_out, x = list_ops.tensor_list_pop_back(
      list_, element_dtype=opts.element_dtype)
  x.set_shape(opts.element_shape)
  return list_out, x


def _py_list_pop(list_, i):
  """Overload of list_pop that executes a Python list append."""
  if i is None:
    x = list_.pop()
  else:
    x = list_.pop(i)
  return list_, x


# TODO(mdan): Look into reducing duplication between all these containers.
class ListStackOpts(
    collections.namedtuple('ListStackOpts',
                           ('element_dtype', 'original_call'))):
  pass


def list_stack(list_, opts):
  """The list stack function.

  This does not have a direct correspondent in Python. The closest idiom to
  this is tf.append or np.stack. It's different from those in the sense that it
  accepts a Tensor list, rather than a list of tensors. It can also accept
  TensorArray. When the target is anything else, the dispatcher will rely on
  ctx.original_call for fallback.

  Args:
    list_: An entity that supports append semantics.
    opts: A ListStackOpts object.

  Returns:
    The output of the stack operation, typically a Tensor.
  """
  assert isinstance(opts, ListStackOpts)

  if isinstance(list_, tensor_array_ops.TensorArray):
    return _tf_tensorarray_stack(list_)
  elif tensor_util.is_tf_type(list_):
    if list_.dtype == dtypes.variant:
      return _tf_tensor_list_stack(list_, opts)
    else:
      # No-op for primitive Tensor arguments.
      return list_
  else:
    return _py_list_stack(list_, opts)


def _tf_tensorarray_stack(list_):
  """Overload of list_stack that stages a TensorArray stack."""
  return list_.stack()


def _tf_tensor_list_stack(list_, opts):
  """Overload of list_stack that stages a Tensor list write."""
  if opts.element_dtype is None:
    raise ValueError('cannot stack a list without knowing its element type;'
                     ' use set_element_type to annotate it')
  return list_ops.tensor_list_stack(list_, element_dtype=opts.element_dtype)


def _py_list_stack(list_, opts):
  """Overload of list_stack that executes a Python list append."""
  # Revert to the original call.
  return opts.original_call(list_)
