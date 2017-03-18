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
# pylint: disable=not-callable
# pylint: disable=redefined-builtin
"""Layers can merge several input tensors into a single output tensor.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine.topology import Layer
from tensorflow.python.framework import tensor_shape


class _Merge(Layer):
  """Generic merge layer for elementwise merge functions.

  Used to implement `Sum`, `Average`, etc.

  Arguments:
      **kwargs: standard layer keyword arguments.
  """

  def __init__(self, **kwargs):
    super(_Merge, self).__init__(**kwargs)
    self.supports_masking = True

  def _merge_function(self, inputs):
    raise NotImplementedError

  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape, list):
      raise ValueError('A merge layer should be called ' 'on a list of inputs.')
    if len(input_shape) < 2:
      raise ValueError('A merge layer should be called '
                       'on a list of at least 2 inputs. '
                       'Got ' + str(len(input_shape)) + ' inputs.')
    if all([shape is None for shape in input_shape]):
      return
    input_shapes = [
        tuple(tensor_shape.TensorShape(shape).as_list())
        for shape in input_shape
    ]
    # TODO(fchollet): handle shapes with None entries.
    input_shapes_set = set(input_shapes)
    if None in input_shapes_set:
      input_shapes_set.remove(None)
    if len(input_shapes_set) > 1:
      raise ValueError('Only tensors of same shape can '
                       'be merged by layer' + self.name +
                       ' Got input shapes: %s' % input_shapes)

  def call(self, inputs):
    return self._merge_function(inputs)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, list):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all([m is None for m in mask]):
      return None
    masks = [K.expand_dims(m, 0) for m in mask if m is not None]
    return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


class Add(_Merge):
  """Layer that adds a list of inputs.

  It takes as input a list of tensors,
  all of the same shape, and returns
  a single tensor (also of the same shape).
  """

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output


class Multiply(_Merge):
  """Layer that multiplies (element-wise) a list of inputs.

  It takes as input a list of tensors,
  all of the same shape, and returns
  a single tensor (also of the same shape).
  """

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output *= inputs[i]
    return output


class Average(_Merge):
  """Layer that averages a list of inputs.

  It takes as input a list of tensors,
  all of the same shape, and returns
  a single tensor (also of the same shape).
  """

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output / len(inputs)


class Maximum(_Merge):
  """Layer that computes the maximum (element-wise) a list of inputs.

  It takes as input a list of tensors,
  all of the same shape, and returns
  a single tensor (also of the same shape).
  """

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = K.maximum(output, inputs[i])
    return output


class Concatenate(_Merge):
  """Layer that concatenates a list of inputs.

  It takes as input a list of tensors,
  all of the same shape expect for the concatenation axis,
  and returns a single tensor, the concatenation of all inputs.

  Arguments:
      axis: Axis along which to concatenate.
      **kwargs: standard layer keyword arguments.
  """

  def __init__(self, axis=-1, **kwargs):
    super(Concatenate, self).__init__(**kwargs)
    self.axis = axis
    self.supports_masking = True

  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape, list):
      raise ValueError('`Concatenate` layer should be called '
                       'on a list of inputs')
    if all([shape is None for shape in input_shape]):
      return
    reduced_inputs_shapes = [
        tensor_shape.TensorShape(shape).as_list() for shape in input_shape
    ]
    shape_set = set()
    for i in range(len(reduced_inputs_shapes)):
      del reduced_inputs_shapes[i][self.axis]
      shape_set.add(tuple(reduced_inputs_shapes[i]))
    if len(shape_set) > 1:
      raise ValueError('`Concatenate` layer requires '
                       'inputs with matching shapes '
                       'except for the concat axis. '
                       'Got inputs shapes: %s' % (input_shape))

  def call(self, inputs):
    if not isinstance(inputs, list):
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of inputs.')
    return K.concatenate(inputs, axis=self.axis)

  def _compute_output_shape(self, input_shape):
    if not isinstance(input_shape, list):
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of inputs.')
    input_shapes = input_shape
    output_shape = tensor_shape.TensorShape(input_shapes[0]).as_list()
    for shape in input_shapes[1:]:
      shape = tensor_shape.TensorShape(shape).as_list()
      if output_shape[self.axis] is None or shape[self.axis] is None:
        output_shape[self.axis] = None
        break
      output_shape[self.axis] += shape[self.axis]
    return tensor_shape.TensorShape(output_shape)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, list):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all([m is None for m in mask]):
      return None
    # Make a list of masks while making sure
    # the dimensionality of each mask
    # is the same as the corresponding input.
    masks = []
    for input_i, mask_i in zip(inputs, mask):
      if mask_i is None:
        # Input is unmasked. Append all 1s to masks,
        # but cast it to uint8 first
        masks.append(K.cast(K.ones_like(input_i), 'uint8'))
      elif K.ndim(mask_i) < K.ndim(input_i):
        # Mask is smaller than the input, expand it
        masks.append(K.expand_dims(mask_i))
      else:
        masks.append(mask_i)
    concatenated = K.concatenate(masks, axis=self.axis)
    return K.all(concatenated, axis=-1, keepdims=False)

  def get_config(self):
    config = {
        'axis': self.axis,
    }
    base_config = super(Concatenate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Dot(_Merge):
  """Layer that computes a dot product between samples in two tensors.

  E.g. if applied to two tensors `a` and `b` of shape `(batch_size, n)`,
  the output will be a tensor of shape `(batch_size, 1)`
  where each entry `i` will be the dot product between
  `a[i]` and `b[i]`.

  Arguments:
      axes: Integer or tuple of integers,
          axis or axes along which to take the dot product.
      normalize: Whether to L2-normalize samples along the
          dot product axis before taking the dot product.
          If set to True, then the output of the dot product
          is the cosine proximity between the two samples.
      **kwargs: Standard layer keyword arguments.
  """

  def __init__(self, axes, normalize=False, **kwargs):
    super(Dot, self).__init__(**kwargs)
    if not isinstance(axes, int):
      if not isinstance(axes, (list, tuple)):
        raise TypeError('Invalid type for `axes` - '
                        'should be a list or an int.')
      if len(axes) != 2:
        raise ValueError('Invalid format for `axes` - '
                         'should contain two elements.')
      if not isinstance(axes[0], int) or not isinstance(axes[1], int):
        raise ValueError('Invalid format for `axes` - '
                         'list elements should be "int".')
    self.axes = axes
    self.normalize = normalize
    self.supports_masking = True

  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape, list) or len(input_shape) != 2:
      raise ValueError('A `Dot` layer should be called '
                       'on a list of 2 inputs.')
    shape1 = tensor_shape.TensorShape(input_shape[0]).as_list()
    shape2 = tensor_shape.TensorShape(input_shape[1]).as_list()
    if shape1 is None or shape2 is None:
      return
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % len(shape1), self.axes % len(shape2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = self.axes
    if shape1[axes[0]] != shape2[axes[1]]:
      raise ValueError('Dimension incompatibility '
                       '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                       'Layer shapes: %s, %s' % (shape1, shape2))

  def call(self, inputs):
    x1 = inputs[0]
    x2 = inputs[1]
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = []
      for i in range(len(self.axes)):
        if self.axes[i] < 0:
          axes.append(self.axes[i] % K.ndim(inputs[i]))
        else:
          axes.append(self.axes[i])
    if self.normalize:
      x1 = K.l2_normalize(x1, axis=axes[0])
      x2 = K.l2_normalize(x2, axis=axes[1])
    output = K.batch_dot(x1, x2, axes)
    return output

  def _compute_output_shape(self, input_shape):
    if not isinstance(input_shape, list) or len(input_shape) != 2:
      raise ValueError('A `Dot` layer should be called '
                       'on a list of 2 inputs.')
    shape1 = tensor_shape.TensorShape(input_shape[0]).as_list()
    shape2 = tensor_shape.TensorShape(input_shape[1]).as_list()
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % len(shape1), self.axes % len(shape2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = self.axes
    shape1.pop(axes[0])
    shape2.pop(axes[1])
    shape2.pop(0)
    output_shape = shape1 + shape2
    if len(output_shape) == 1:
      output_shape += [1]
    return tensor_shape.TensorShape(output_shape)

  def compute_mask(self, inputs, mask=None):
    return None

  def get_config(self):
    config = {
        'axes': self.axes,
        'normalize': self.normalize,
    }
    base_config = super(Dot, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def add(inputs, **kwargs):
  """Functional interface to the `Add` layer.

  Arguments:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the sum of the inputs.
  """
  return Add(**kwargs)(inputs)


def multiply(inputs, **kwargs):
  """Functional interface to the `Multiply` layer.

  Arguments:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the element-wise product of the inputs.
  """
  return Multiply(**kwargs)(inputs)


def average(inputs, **kwargs):
  """Functional interface to the `Average` layer.

  Arguments:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the average of the inputs.
  """
  return Average(**kwargs)(inputs)


def maximum(inputs, **kwargs):
  """Functional interface to the `Maximum` layer.

  Arguments:
      inputs: A list of input tensors (at least 2).
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the element-wise maximum of the inputs.
  """
  return Maximum(**kwargs)(inputs)


def concatenate(inputs, axis=-1, **kwargs):
  """Functional interface to the `Concatenate` layer.

  Arguments:
      inputs: A list of input tensors (at least 2).
      axis: Concatenation axis.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the concatenation of the inputs alongside axis `axis`.
  """
  return Concatenate(axis=axis, **kwargs)(inputs)


def dot(inputs, axes, normalize=False, **kwargs):
  """Functional interface to the `Dot` layer.

  Arguments:
      inputs: A list of input tensors (at least 2).
      axes: Integer or tuple of integers,
          axis or axes along which to take the dot product.
      normalize: Whether to L2-normalize samples along the
          dot product axis before taking the dot product.
          If set to True, then the output of the dot product
          is the cosine proximity between the two samples.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the dot product of the samples from the inputs.
  """
  return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)
