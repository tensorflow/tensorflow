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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


@tf_export("clip_by_value")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def clip_by_value(t, clip_value_min, clip_value_max,
                  name=None):
  """Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
  correct results.

  For example:

  Basic usage passes a scalar as the min and max value.

  >>> t = tf.constant([[-10., -1., 0.], [0., 2., 10.]])
  >>> t2 = tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1)
  >>> t2.numpy()
  array([[-1., -1.,  0.],
         [ 0.,  1.,  1.]], dtype=float32)

  The min and max can be the same size as `t`, or broadcastable to that size.

  >>> t = tf.constant([[-1, 0., 10.], [-1, 0, 10]])
  >>> clip_min = [[2],[1]]
  >>> t3 = tf.clip_by_value(t, clip_value_min=clip_min, clip_value_max=100)
  >>> t3.numpy()
  array([[ 2.,  2., 10.],
         [ 1.,  1., 10.]], dtype=float32)

  Broadcasting fails, intentionally, if you would expand the dimensions of `t`

  >>> t = tf.constant([[-1, 0., 10.], [-1, 0, 10]])
  >>> clip_min = [[[2, 1]]] # Has a third axis
  >>> t4 = tf.clip_by_value(t, clip_value_min=clip_min, clip_value_max=100)
  Traceback (most recent call last):
  ...
  InvalidArgumentError: Incompatible shapes: [2,3] vs. [1,1,2]

  It throws a `TypeError` if you try to clip an `int` to a `float` value
  (`tf.cast` the input to `float` first).

  >>> t = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
  >>> t5 = tf.clip_by_value(t, clip_value_min=-3.1, clip_value_max=3.1)
  Traceback (most recent call last):
  ...
  TypeError: Cannot convert ...


  Args:
    t: A `Tensor` or `IndexedSlices`.
    clip_value_min: The minimum value to clip to. A scalar `Tensor` or one that
      is broadcastable to the shape of `t`.
    clip_value_max: The maximum value to clip to. A scalar `Tensor` or one that
      is broadcastable to the shape of `t`.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor` or `IndexedSlices`.

  Raises:
    `tf.errors.InvalidArgumentError`: If the clip tensors would trigger array
      broadcasting that would make the returned tensor larger than the input.
    TypeError: If dtype of the input is `int32` and dtype of
      the `clip_value_min` or `clip_value_max` is `float32`
  """
  with ops.name_scope(name, "clip_by_value",
                      [t, clip_value_min, clip_value_max]) as name:
    values = ops.convert_to_tensor(
        t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
        name="t")

    # Go through list of tensors, for each value in each tensor clip
    t_min = math_ops.minimum(values, clip_value_max)
    # Assert that the shape is compatible with the initial shape,
    # to prevent unintentional broadcasting.
    values.shape.assert_is_compatible_with(t_min.shape)

    t_max = math_ops.maximum(t_min, clip_value_min, name=name)
    values.shape.assert_is_compatible_with(t_max.shape)

    if isinstance(t, indexed_slices.IndexedSlices):
      t_max = indexed_slices.IndexedSlices(t_max, t.indices, t.dense_shape)

  return t_max
  # TODO(scottzhu): switch to use new implementation in 2 weeks.
  # return gen_math_ops.clip_by_value(
  #     t, clip_value_min, clip_value_max, name=name)


@ops.RegisterGradient("ClipByValue")
def _clip_by_value_grad(op, grad):
  """Returns grad of clip_by_value."""
  x = op.inputs[0]
  y = op.inputs[1]
  z = op.inputs[2]
  gdtype = grad.dtype
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  sz = array_ops.shape(z)
  gradshape = array_ops.shape(grad)
  zeros = array_ops.zeros(gradshape, gdtype)
  xymask = math_ops.less(x, y)
  xzmask = math_ops.greater(x, z)
  _, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
  _, rz = gen_array_ops.broadcast_gradient_args(sx, sz)
  xgrad = array_ops.where(math_ops.logical_or(xymask, xzmask), zeros, grad)
  ygrad = array_ops.where(xymask, grad, zeros)
  zgrad = array_ops.where(xzmask, grad, zeros)
  gy = array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy)
  gz = array_ops.reshape(math_ops.reduce_sum(zgrad, rz), sz)
  return xgrad, gy, gz


@tf_export("clip_by_norm")
@dispatch.add_dispatch_support
def clip_by_norm(t, clip_norm, axes=None, name=None):
  """Clips tensor values to a maximum L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
  along the dimensions given in `axes`. Specifically, in the default case
  where all dimensions are used for calculation, if the L2-norm of `t` is
  already less than or equal to `clip_norm`, then `t` is not modified. If
  the L2-norm is greater than `clip_norm`, then this operation returns a
  tensor of the same type and shape as `t` with its values set to:

  `t * clip_norm / l2norm(t)`

  In this case, the L2-norm of the output tensor is `clip_norm`.

  As another example, if `t` is a matrix and `axes == [1]`, then each row
  of the output will have L2-norm less than or equal to `clip_norm`. If
  `axes == [0]` instead, each column of the output will be clipped.

  Code example:

  >>> some_nums = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float32)
  >>> tf.clip_by_norm(some_nums, 2.0).numpy()
  array([[0.26967996, 0.5393599 , 0.80903983, 1.0787199 , 1.3483998 ]],
        dtype=float32)

  This operation is typically used to clip gradients before applying them with
  an optimizer.  Most gradient data is a collection of different shaped tensors
  for different parts of the model.  Thus, this is a common usage:

  ```
  # Get your gradients after training
  loss_value, grads = grad(model, features, labels)

  # Apply some clipping
  grads = [tf.clip_by_norm(g, norm)
               for g in grads]

  # Continue on with training
  optimizer.apply_gradients(grads)
  ```

  Args:
    t: A `Tensor` or `IndexedSlices`.  This must be a floating point type.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value, also
      floating point.
      Note: If a negative clip_norm is provided, it will be treated as zero.
    axes: A 1-D (vector) `Tensor` of type int32 containing the dimensions to use
      for computing the L2-norm. If `None` (the default), uses all dimensions.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor` or `IndexedSlices`.

  Raises:
    ValueError: If the clip_norm tensor is not a 0-D scalar tensor.
    TypeError: If dtype of the input is not a floating point or
      complex type.
  """
  with ops.name_scope(name, "clip_by_norm", [t, clip_norm]) as name:
    values = ops.convert_to_tensor(
        t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
        name="t")

    if np.isscalar(clip_norm):
      if clip_norm < 0:
        clip_norm = 0
    else:
      clip_norm = math_ops.cast(
          math_ops.maximum(clip_norm, 0), dtype=values.dtype
      )

    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    l2sum = math_ops.reduce_sum(values * values, axes, keepdims=True)
    pred = l2sum > 0
    # Two-tap tf.where trick to bypass NaN gradients
    l2sum_safe = array_ops.where(pred, l2sum, array_ops.ones_like(l2sum))
    l2norm = array_ops.where(pred, math_ops.sqrt(l2sum_safe), l2sum)
    intermediate = values * clip_norm
    # Assert that the shape is compatible with the initial shape,
    # to prevent unintentional broadcasting.
    values.shape.assert_is_compatible_with(intermediate.shape)
    values_clip = array_ops.identity(
        intermediate / math_ops.maximum(l2norm, clip_norm), name=name)

    if isinstance(t, indexed_slices.IndexedSlices):
      return indexed_slices.IndexedSlices(values_clip, t.indices, t.dense_shape)

    return values_clip


@tf_export("linalg.global_norm", v1=["linalg.global_norm", "global_norm"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("global_norm")
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
  if (not isinstance(t_list, collections_abc.Sequence) or
      isinstance(t_list, str)):
    raise TypeError("`t_list` should be a sequence of tensors. Received "
                    f"{type(t_list)}.")
  t_list = list(t_list)
  with ops.name_scope(name, "global_norm", t_list) as name:
    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
            name="t_%d" % i) if t is not None else t
        for i, t in enumerate(t_list)
    ]
    half_squared_norms = []
    for v in values:
      if v is not None:
        with ops.colocate_with(v):
          half_squared_norms.append(gen_nn_ops.l2_loss(v))

    half_squared_norm = math_ops.reduce_sum(
        array_ops_stack.stack(half_squared_norms))

    norm = math_ops.sqrt(
        half_squared_norm *
        constant_op.constant(2.0, dtype=half_squared_norm.dtype),
        name="global_norm")

  return norm


@tf_export("clip_by_global_norm")
@dispatch.add_dispatch_support
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

  If `global_norm == infinity` then the entries in `t_list` are all set to `NaN`
  to signal that an error occurred.

  Any of the entries of `t_list` that are of type `None` are ignored.

  This is the correct way to perform gradient clipping (Pascanu et al., 2012).

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

  References:
    On the difficulty of training Recurrent Neural Networks:
      [Pascanu et al., 2012](http://proceedings.mlr.press/v28/pascanu13.html)
      ([pdf](http://proceedings.mlr.press/v28/pascanu13.pdf))
  """
  if (not isinstance(t_list, collections_abc.Sequence) or
      isinstance(t_list, str)):
    raise TypeError("`t_list` should be a sequence of tensors. Received "
                    f"{type(t_list)}.")
  t_list = list(t_list)
  if use_norm is None:
    use_norm = global_norm(t_list, name)

  with ops.name_scope(name, "clip_by_global_norm",
                      t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale_for_finite = clip_norm * math_ops.minimum(
        1.0 / use_norm,
        constant_op.constant(1.0, dtype=use_norm.dtype) / clip_norm)
    # If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
    # this will make scale NaN.
    scale = scale_for_finite + (use_norm - use_norm)

    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
            name="t_%d" % i) if t is not None else t
        for i, t in enumerate(t_list)
    ]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with ops.colocate_with(v):
          values_clipped.append(
              array_ops.identity(
                  v * math_ops.cast(scale, v.dtype), name="%s_%d" % (name, i)
              )
          )

    list_clipped = [
        indexed_slices.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, indexed_slices.IndexedSlices) else c_v
        for (c_v, t) in zip(values_clipped, t_list)
    ]

  return list_clipped, use_norm


@deprecation.deprecated(
    date=None,
    instructions="clip_by_average_norm is deprecated in TensorFlow 2.0. Please "
    "use clip_by_norm(t, clip_norm * tf.cast(tf.size(t), tf.float32), name) "
    "instead.")
@tf_export(v1=["clip_by_average_norm"])
@dispatch.add_dispatch_support
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
  with ops.name_scope(name, "clip_by_average_norm", [t, clip_norm]) as name:
    t = ops.convert_to_tensor(t, name="t")

    # Calculate L2-norm per element, clip elements by ratio of clip_norm to
    # L2-norm per element
    n_element = math_ops.cast(array_ops.size(t), dtypes.float32)
    l2norm_inv = math_ops.rsqrt(
        math_ops.reduce_sum(t * t, math_ops.range(array_ops.rank(t))))
    tclip = array_ops.identity(
        t * clip_norm * math_ops.minimum(
            l2norm_inv * n_element, constant_op.constant(1.0) / clip_norm),
        name=name)

  return tclip
