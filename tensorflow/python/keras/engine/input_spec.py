# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Contains the InputSpec class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util.tf_export import tf_export


@keras_export('keras.layers.InputSpec')
@tf_export(v1=['layers.InputSpec'])
class InputSpec(object):
  """Specifies the rank, dtype and shape of every input to a layer.

  Layers can expose (if appropriate) an `input_spec` attribute:
  an instance of `InputSpec`, or a nested structure of `InputSpec` instances
  (one per input tensor). These objects enable the layer to run input
  compatibility checks for input structure, input rank, input shape, and
  input dtype.

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.

  Arguments:
      dtype: Expected DataType of the input.
      shape: Shape tuple, expected shape of the input
          (may include None for unchecked axes).
      ndim: Integer, expected rank of the input.
      max_ndim: Integer, maximum rank of the input.
      min_ndim: Integer, minimum rank of the input.
      axes: Dictionary mapping integer axes to
          a specific dimension value.
  """

  def __init__(self,
               dtype=None,
               shape=None,
               ndim=None,
               max_ndim=None,
               min_ndim=None,
               axes=None):
    self.dtype = dtypes.as_dtype(dtype).name if dtype is not None else None
    if shape is not None:
      self.ndim = len(shape)
      self.shape = shape
    else:
      self.ndim = ndim
      self.shape = None
    self.max_ndim = max_ndim
    self.min_ndim = min_ndim
    try:
      axes = axes or {}
      self.axes = {int(k): axes[k] for k in axes}
    except (ValueError, TypeError):
      raise TypeError('The keys in axes must be integers.')

    if self.axes and (self.ndim is not None or self.max_ndim is not None):
      max_dim = (self.ndim if self.ndim else self.max_ndim) - 1
      max_axis = max(self.axes)
      if max_axis > max_dim:
        raise ValueError('Axis {} is greater than the maximum allowed value: {}'
                         .format(max_axis, max_dim))

  def __repr__(self):
    spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
            ('shape=' + str(self.shape)) if self.shape else '',
            ('ndim=' + str(self.ndim)) if self.ndim else '',
            ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
            ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
            ('axes=' + str(self.axes)) if self.axes else '']
    return 'InputSpec(%s)' % ', '.join(x for x in spec if x)

  def get_config(self):
    return {
        'dtype': self.dtype,
        'shape': self.shape,
        'ndim': self.ndim,
        'max_ndim': self.max_ndim,
        'min_ndim': self.min_ndim,
        'axes': self.axes}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def to_tensor_shape(spec):
  """Returns a tf.TensorShape object that matches the shape specifications.

  If the InputSpec's shape or ndim is defined, this method will return a fully
  or partially-known shape. Otherwise, the returned TensorShape is None.

  Args:
    spec: an InputSpec object.

  Returns:
    a tf.TensorShape object
  """
  if spec.ndim is None and spec.shape is None:
    return tensor_shape.TensorShape(None)
  elif spec.shape is not None:
    return tensor_shape.TensorShape(spec.shape)
  else:
    shape = [None] * spec.ndim
    for a in spec.axes:
      shape[a] = spec.axes[a]  # Assume that axes is defined
    return tensor_shape.TensorShape(shape)


def assert_input_compatibility(input_spec, inputs, layer_name):
  """Checks compatibility between the layer and provided inputs.

  This checks that the tensor(s) `inputs` verify the input assumptions
  of a layer (if any). If not, a clear and actional exception gets raised.

  Arguments:
      input_spec: An InputSpec instance, list of InputSpec instances, a nested
          structure of InputSpec instances, or None.
      inputs: Input tensor, list of input tensors, or a nested structure of
          input tensors.
      layer_name: String, name of the layer (for error message formatting).

  Raises:
      ValueError: in case of mismatch between
          the provided inputs and the expectations of the layer.
  """
  if not input_spec:
    return

  inputs = nest.flatten(inputs)
  input_spec = nest.flatten(input_spec)
  if len(inputs) != len(input_spec):
    raise ValueError('Layer ' + layer_name + ' expects ' +
                     str(len(input_spec)) + ' inputs, '
                     'but it received ' + str(len(inputs)) +
                     ' input tensors. Inputs received: ' + str(inputs))
  for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
    if spec is None:
      continue

    if (spec.ndim is not None or
        spec.min_ndim is not None or
        spec.max_ndim is not None):
      if x.shape.ndims is None:
        raise ValueError('Input ' + str(input_index) + ' of layer ' +
                         layer_name + ' is incompatible with the layer: '
                         'its rank is undefined, but the layer requires a '
                         'defined rank.')

    # Check ndim.
    if spec.ndim is not None:
      ndim = x.shape.ndims
      if ndim != spec.ndim:
        raise ValueError('Input ' + str(input_index) + ' of layer ' +
                         layer_name + ' is incompatible with the layer: '
                         'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                         str(ndim) + '. Full shape received: ' +
                         str(x.shape.as_list()))
    if spec.max_ndim is not None:
      ndim = x.shape.ndims
      if ndim is not None and ndim > spec.max_ndim:
        raise ValueError('Input ' + str(input_index) + ' of layer ' +
                         layer_name + ' is incompatible with the layer: '
                         'expected max_ndim=' + str(spec.max_ndim) +
                         ', found ndim=' + str(ndim))
    if spec.min_ndim is not None:
      ndim = x.shape.ndims
      if ndim is not None and ndim < spec.min_ndim:
        raise ValueError('Input ' + str(input_index) + ' of layer ' +
                         layer_name + ' is incompatible with the layer: '
                         ': expected min_ndim=' + str(spec.min_ndim) +
                         ', found ndim=' + str(ndim) +
                         '. Full shape received: ' +
                         str(x.shape.as_list()))
    # Check dtype.
    if spec.dtype is not None:
      if x.dtype != spec.dtype:
        raise ValueError('Input ' + str(input_index) + ' of layer ' +
                         layer_name + ' is incompatible with the layer: '
                         'expected dtype=' + str(spec.dtype) +
                         ', found dtype=' + str(x.dtype))
    # Check specific shape axes.
    if spec.axes:
      shape = x.shape.as_list()
      if shape is not None:
        for axis, value in spec.axes.items():
          if hasattr(value, 'value'):
            value = value.value
          if value is not None and shape[int(axis)] not in {value, None}:
            raise ValueError(
                'Input ' + str(input_index) + ' of layer ' + layer_name + ' is'
                ' incompatible with the layer: expected axis ' + str(axis) +
                ' of input shape to have value ' + str(value) +
                ' but received input with shape ' + str(shape))
    # Check shape.
    if spec.shape is not None:
      shape = x.shape.as_list()
      if shape is not None:
        for spec_dim, dim in zip(spec.shape, shape):
          if spec_dim is not None and dim is not None:
            if spec_dim != dim:
              raise ValueError('Input ' + str(input_index) +
                               ' is incompatible with layer ' + layer_name +
                               ': expected shape=' + str(spec.shape) +
                               ', found shape=' + str(shape))


def to_tensor_spec(input_spec, default_dtype=None):
  """Converts a Keras InputSpec object to a TensorSpec."""
  default_dtype = default_dtype or backend.floatx()
  if isinstance(input_spec, InputSpec):
    dtype = input_spec.dtype or default_dtype
    return tensor_spec.TensorSpec(to_tensor_shape(input_spec), dtype)
  return tensor_spec.TensorSpec(None, default_dtype)
