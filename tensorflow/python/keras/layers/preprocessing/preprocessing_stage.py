# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Preprocessing stage."""
# pylint: disable=g-classes-have-attributes

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest


# Sequential methods should take precedence.
class PreprocessingStage(sequential.Sequential,
                         base_preprocessing_layer.PreprocessingLayer):
  """A sequential preprocessing stage.

  This preprocessing stage wraps a list of preprocessing layers into a
  Sequential-like object that enables you to `adapt()` the whole list via
  a single `adapt()` call on the preprocessing stage.

  Args:
    layers: List of layers. Can include layers that aren't preprocessing layers.
    name: String. Optional name for the preprocessing stage object.
  """

  def adapt(self, data, reset_state=True):
    """Adapt the state of the layers of the preprocessing stage to the data.

    Args:
      data: A batched Dataset object, or a NumPy array, or an EagerTensor.
        Data to be iterated over to adapt the state of the layers in this
        preprocessing stage.
      reset_state: Whether this call to `adapt` should reset the state of
        the layers in this preprocessing stage.
    """
    if not isinstance(data,
                      (dataset_ops.DatasetV2, np.ndarray, ops.EagerTensor)):
      raise ValueError(
          '`adapt()` requires a batched Dataset, an EagerTensor, '
          'or a Numpy array as input, '
          'got {}'.format(type(data)))
    if isinstance(data, dataset_ops.DatasetV2):
      # Validate the datasets to try and ensure we haven't been passed one with
      # infinite size. That would cause an infinite loop here.
      if tf_utils.dataset_is_infinite(data):
        raise ValueError(
            'The dataset passed to `adapt()` has an infinite number of '
            'elements. Please use dataset.take(...) to make the number '
            'of elements finite.')

    for current_layer_index in range(0, len(self.layers)):
      if not hasattr(self.layers[current_layer_index], 'adapt'):
        # Skip any layer that does not need adapting.
        continue

      def map_fn(x):
        """Maps `PreprocessingStage` inputs to inputs at `current_layer_index`.

        Args:
          x: Batch of inputs seen in entry of the `PreprocessingStage` instance.

        Returns:
          Batch of inputs to be processed by layer
            `self.layers[current_layer_index]`
        """
        if current_layer_index == 0:  # pylint: disable=cell-var-from-loop
          return x
        for i in range(current_layer_index):  # pylint: disable=cell-var-from-loop
          x = self.layers[i](x)
        return x

      if isinstance(data, dataset_ops.DatasetV2):
        current_layer_data = data.map(map_fn)
      else:
        current_layer_data = map_fn(data)
      self.layers[current_layer_index].adapt(current_layer_data,
                                             reset_state=reset_state)


# Functional methods shoud take precedence.
class FunctionalPreprocessingStage(functional.Functional,
                                   base_preprocessing_layer.PreprocessingLayer):
  """A functional preprocessing stage.

  This preprocessing stage wraps a graph of preprocessing layers into a
  Functional-like object that enables you to `adapt()` the whole graph via
  a single `adapt()` call on the preprocessing stage.

  Preprocessing stage is not a complete model, so it cannot be called with
  `fit()`. However, it is possible to add regular layers that may be trainable
  to a preprocessing stage.

  A functional preprocessing stage is created in the same way as `Functional`
  models. A stage can be instantiated by passing two arguments to
  `__init__`. The first argument is the `keras.Input` Tensors that represent
  the inputs to the stage. The second argument specifies the output
  tensors that represent the outputs of this stage. Both arguments can be a
  nested structure of tensors.

  Example:

  >>> inputs = {'x2': tf.keras.Input(shape=(5,)),
  ...           'x1': tf.keras.Input(shape=(1,))}
  >>> norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
  >>> y = norm_layer(inputs['x2'])
  >>> y, z = tf.keras.layers.Lambda(lambda x: (x, x))(inputs['x1'])
  >>> outputs = [inputs['x1'], [y, z]]
  >>> stage = FunctionalPreprocessingStage(inputs, outputs)

  Args:
    inputs: An input tensor (must be created via `tf.keras.Input()`), or a list,
      a dict, or a nested strcture of input tensors.
    outputs: An output tensor, or a list, a dict or a nested structure of output
      tensors.
    name: String, optional. Name of the preprocessing stage.
  """

  def fit(self, *args, **kwargs):
    raise ValueError(
        'Preprocessing stage is not a complete model, and hence should not be '
        '`fit`. Instead, you may feed data to `adapt` the stage to set '
        'appropriate states of the layers in the stage.')

  def adapt(self, data, reset_state=True):
    """Adapt the state of the layers of the preprocessing stage to the data.

    Args:
      data: A batched Dataset object, a NumPy array, an EagerTensor, or a list,
        dict or nested structure of Numpy Arrays or EagerTensors. The elements
        of Dataset object need to conform with inputs of the stage. The first
        dimension of NumPy arrays or EagerTensors are understood to be batch
        dimension. Data to be iterated over to adapt the state of the layers in
        this preprocessing stage.
      reset_state: Whether this call to `adapt` should reset the state of the
        layers in this preprocessing stage.

    Examples:

    >>> # For a stage with dict input
    >>> inputs = {'x2': tf.keras.Input(shape=(5,)),
    ...           'x1': tf.keras.Input(shape=(1,))}
    >>> outputs = [inputs['x1'], inputs['x2']]
    >>> stage = FunctionalPreprocessingStage(inputs, outputs)
    >>> ds = tf.data.Dataset.from_tensor_slices({'x1': tf.ones((4,5)),
    ...                                          'x2': tf.ones((4,1))})
    >>> sorted(ds.element_spec.items()) # Check element_spec
    [('x1', TensorSpec(shape=(5,), dtype=tf.float32, name=None)),
     ('x2', TensorSpec(shape=(1,), dtype=tf.float32, name=None))]
    >>> stage.adapt(ds)
    >>> data_np = {'x1': np.ones((4, 5)), 'x2': np.ones((4, 1))}
    >>> stage.adapt(data_np)

    """
    if not isinstance(data, dataset_ops.Dataset):
      data = self._flatten_to_reference_inputs(data)
      if any(not isinstance(datum, (np.ndarray, ops.EagerTensor))
             for datum in data):
        raise ValueError(
            '`adapt()` requires a batched Dataset, a list of EagerTensors '
            'or Numpy arrays as input, got {}'.format(type(data)))
      ds_input = [
          dataset_ops.Dataset.from_tensor_slices(x).batch(1) for x in data
      ]

    if isinstance(data, dataset_ops.Dataset):
      # Validate the datasets to try and ensure we haven't been passed one with
      # infinite size. That would cause an infinite loop here.
      if tf_utils.dataset_is_infinite(data):
        raise ValueError(
            'The dataset passed to `adapt()` has an infinite number of '
            'elements. Please use dataset.take(...) to make the number '
            'of elements finite.')
      # Unzip dataset object to a list of single input dataset.
      ds_input = _unzip_dataset(data)

    # Dictionary mapping reference tensors to datasets
    ds_dict = {}
    tensor_usage_count = self._tensor_usage_count
    for x, y in zip(self.inputs, ds_input):
      x_id = str(id(x))
      ds_dict[x_id] = [y] * tensor_usage_count[x_id]

    nodes_by_depth = self._nodes_by_depth
    depth_keys = sorted(nodes_by_depth.keys(), reverse=True)

    def build_map_fn(node, args, kwargs):
      if not isinstance(args.element_spec, tuple):

        def map_fn(*x):
          return nest.flatten(node.layer(*x, **kwargs))
      else:

        def map_fn(*x):
          return nest.flatten(node.layer(x, **kwargs))

      return map_fn

    for depth in depth_keys:
      for node in nodes_by_depth[depth]:
        # Input node
        if node.is_input:
          continue

        # Node with input not computed yet
        if any(t_id not in ds_dict for t_id in node.flat_input_ids):
          continue

        args, kwargs = node.map_arguments(ds_dict)
        args = dataset_ops.Dataset.zip(nest.list_to_tuple(*args))

        if hasattr(node.layer, 'adapt'):
          node.layer.adapt(args, reset_state=reset_state)

        map_fn = build_map_fn(node, args, kwargs)
        outputs = args.map(map_fn)
        outputs = _unzip_dataset(outputs)

        # Update ds_dict.
        for x_id, y in zip(node.flat_output_ids, outputs):
          ds_dict[x_id] = [y] * tensor_usage_count[x_id]


def _unzip_dataset(ds):
  """Unzip dataset into a list of single element datasets.

  Args:
    ds: A Dataset object.

  Returns:
    A list of Dataset object, each correspond to one of the `element_spec` of
    the input Dataset object.

  Example:

  >>> ds1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> ds2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
  >>> ds_zipped_tuple = tf.data.Dataset.zip((ds1, ds2))
  >>> ds_unzipped_tuple = _unzip_dataset(ds_zipped_tuple)
  >>> ds_zipped_dict = tf.data.Dataset.zip({'ds1': ds1, 'ds2': ds2})
  >>> ds_unzipped_dict = _unzip_dataset(ds_zipped_dict)

  Then the two elements of `ds_unzipped_tuple` and `ds_unzipped_dict` are both
  the same as `ds1` and `ds2`.
  """
  element_count = len(nest.flatten(ds.element_spec))
  ds_unzipped = []
  for i in range(element_count):

    def map_fn(*x, j=i):
      return nest.flatten(x)[j]

    ds_unzipped.append(ds.map(map_fn))
  return ds_unzipped
