# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Operations for clipping (gradient, weight) tensors to min/max values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def clip_by_value(t, clip_value_min, clip_value_max,
                  name=None):
  """Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Args:
    t: A `Tensor`.
    clip_value_min: A 0-D (scalar) `Tensor`. The minimum value to clip by.
    clip_value_max: A 0-D (scalar) `Tensor`. The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
  with ops.op_scope([t, clip_value_min, clip_value_max], name,
                   "clip_by_value") as name:
    t = ops.convert_to_tensor(t, name="t")

    # Go through list of tensors, for each value in each tensor clip
    t_min = math_ops.minimum(t, clip_value_max)
    t_max = math_ops.maximum(t_min, clip_value_min, name=name)

  return t_max


def clip_by_norm(t, clip_norm, name=None):
  """Clips tensor values to a maximum L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`.
  Specifically, if the L2-norm is already less than or equal to `clip_norm`,
  then `t` is not modified. If the L2-norm is greater than `clip_norm`, then
  this operation returns a tensor of the same type and shape as `t` with its
  values set to:

  `t * clip_norm / l2norm(t)`

  In this case, the L2-norm of the output tensor is `clip_norm`.

  This operation is typically used to clip gradients before applying them with
  an optimizer.

  Args:
    t: A `Tensor`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
  with ops.op_scope([t, clip_norm], name, "clip_by_norm") as name:
    t = ops.convert_to_tensor(t, name="t")

    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    l2norm_inv = math_ops.rsqrt(
        math_ops.reduce_sum(t * t, math_ops.range(array_ops.rank(t))))
    tclip = array_ops.identity(t * clip_norm * math_ops.minimum(
        l2norm_inv, constant_op.constant(1.0 / clip_norm)), name=name)

  return tclip

def global_norm(t_list, name=None):
  """Computes the global norm of multiple tensors.

  Given a tuple or list of tensors `t_list`, this operation returns the
  global norm of the elements in all tensors in `t_list`. The global norm is
  computed as:

  `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`

  Any entries in `t_list` that are of type None are ignored.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    name: A name for the operation (optional).

  Returns:
    A 0-D (scalar) `Tensor` of type `float`.

  Raises:
    TypeError: If `t_list` is not a sequence.
  """
  if (not isinstance(t_list, collections.Sequence)
      or isinstance(t_list, six.string_types)):
    raise TypeError("t_list should be a sequence")
  t_list = list(t_list)
  with ops.op_scope(t_list, name, "global_norm") as name:
    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, ops.IndexedSlices) else t,
            name="t_%d" % i)
        if t is not None else t
        for i, t in enumerate(t_list)]
    half_squared_norms = []
    for v in values:
      if v is not None:
        with ops.colocate_with(v):
          half_squared_norms.append(nn_ops.l2_loss(v))

    half_squared_norm = math_ops.reduce_sum(array_ops.pack(half_squared_norms))

    norm = math_ops.sqrt(
        half_squared_norm *
        constant_op.constant(2.0, dtype=half_squared_norm.dtype),
        name="global_norm")

  return norm

def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
  """Clips values of multiple tensors by the ratio of the sum of their norms.

  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
  if you've already computed the global norm for `t_list`, you can specify
  the global norm with `use_norm`.

  To perform the clipping, the values `t_list[i]` are set to:

      t_list[i] * clip_norm / max(global_norm, clip_norm)

  where:

      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.

  Any of the entries of `t_list` that are of type `None` are ignored.

  This is the correct way to perform gradient clipping (for example, see
  [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).

  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.

  Raises:
    TypeError: If `t_list` is not a sequence.
  """
  if (not isinstance(t_list, collections.Sequence)
      or isinstance(t_list, six.string_types)):
    raise TypeError("t_list should be a sequence")
  t_list = list(t_list)
  if use_norm is None:
    use_norm = global_norm(t_list, name)

  with ops.op_scope(t_list + [clip_norm], name, "clip_by_global_norm") as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale = clip_norm * math_ops.minimum(
        1.0 / use_norm,
        constant_op.constant(1.0 / clip_norm, dtype=use_norm.dtype))

    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, ops.IndexedSlices) else t,
            name="t_%d" % i)
        if t is not None else t
        for i, t in enumerate(t_list)]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with ops.colocate_with(v):
          values_clipped.append(
              array_ops.identity(v * scale, name="%s_%d" % (name, i)))

    list_clipped = [
        ops.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, ops.IndexedSlices)
        else c_v
        for (c_v, t) in zip(values_clipped, t_list)]

  return list_clipped, use_norm


def clip_by_average_norm(t, clip_norm, name=None):
  """Clips tensor values to a maximum average L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its average L2-norm is less than or equal to
  `clip_norm`. Specifically, if the average L2-norm is already less than or
  equal to `clip_norm`, then `t` is not modified. If the average L2-norm is
  greater than `clip_norm`, then this operation returns a tensor of the same
  type and shape as `t` with its values set to:

  `t * clip_norm / l2norm_avg(t)`

  In this case, the average L2-norm of the output tensor is `clip_norm`.

  This operation is typically used to clip gradients before applying them with
  an optimizer.

  Args:
    t: A `Tensor`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
  with ops.op_scope([t, clip_norm], name, "clip_by_average_norm") as name:
    t = ops.convert_to_tensor(t, name="t")

    # Calculate L2-norm per element, clip elements by ratio of clip_norm to
    # L2-norm per element
    n_element = math_ops.cast(array_ops.size(t), dtypes.float32)
    l2norm_inv = math_ops.rsqrt(
        math_ops.reduce_sum(t * t, math_ops.range(array_ops.rank(t))))
    tclip = array_ops.identity(
        t * clip_norm * math_ops.minimum(
            l2norm_inv * n_element, constant_op.constant(1.0 / clip_norm)),
        name=name)

  return tclip
