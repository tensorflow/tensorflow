# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""This API defines FeatureColumn abstraction."""

# This file was originally under tf/python/feature_column, and was moved to
# Keras package in order to remove the reverse dependency from TF to Keras.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


class _BaseFeaturesLayer(Layer):
  """Base class for DenseFeatures and SequenceFeatures.

  Defines common methods and helpers.

  Args:
    feature_columns: An iterable containing the FeatureColumns to use as
      inputs to your model.
    expected_column_type: Expected class for provided feature columns.
    trainable:  Boolean, whether the layer's variables will be updated via
      gradient descent during training.
    name: Name to give to the DenseFeatures.
    **kwargs: Keyword arguments to construct a layer.

  Raises:
    ValueError: if an item in `feature_columns` doesn't match
      `expected_column_type`.
  """

  def __init__(self,
               feature_columns,
               expected_column_type,
               trainable,
               name,
               partitioner=None,
               **kwargs):
    super(_BaseFeaturesLayer, self).__init__(
        name=name, trainable=trainable, **kwargs)
    self._feature_columns = feature_column_v2._normalize_feature_columns(  # pylint: disable=protected-access
        feature_columns)
    self._state_manager = feature_column_v2._StateManagerImpl(  # pylint: disable=protected-access
        self, self.trainable)
    self._partitioner = partitioner
    for column in self._feature_columns:
      if not isinstance(column, expected_column_type):
        raise ValueError(
            'Items of feature_columns must be a {}. '
            'You can wrap a categorical column with an '
            'embedding_column or indicator_column. Given: {}'.format(
                expected_column_type, column))

  def build(self, _):
    for column in self._feature_columns:
      with variable_scope._pure_variable_scope(  # pylint: disable=protected-access
          self.name,
          partitioner=self._partitioner):
        with variable_scope._pure_variable_scope(  # pylint: disable=protected-access
            feature_column_v2._sanitize_column_name_for_variable_scope(  # pylint: disable=protected-access
                column.name)):
          column.create_state(self._state_manager)
    super(_BaseFeaturesLayer, self).build(None)

  def _output_shape(self, input_shape, num_elements):
    """Computes expected output shape of the layer or a column's dense tensor.

    Args:
      input_shape: Tensor or array with batch shape.
      num_elements: Size of the last dimension of the output.

    Returns:
      Tuple with output shape.
    """
    raise NotImplementedError('Calling an abstract method.')

  def compute_output_shape(self, input_shape):
    total_elements = 0
    for column in self._feature_columns:
      total_elements += column.variable_shape.num_elements()
    return self._target_shape(input_shape, total_elements)

  def _process_dense_tensor(self, column, tensor):
    """Reshapes the dense tensor output of a column based on expected shape.

    Args:
      column: A DenseColumn or SequenceDenseColumn object.
      tensor: A dense tensor obtained from the same column.

    Returns:
      Reshaped dense tensor.
    """
    num_elements = column.variable_shape.num_elements()
    target_shape = self._target_shape(array_ops.shape(tensor), num_elements)
    return array_ops.reshape(tensor, shape=target_shape)

  def _verify_and_concat_tensors(self, output_tensors):
    """Verifies and concatenates the dense output of several columns."""
    feature_column_v2._verify_static_batch_size_equality(  # pylint: disable=protected-access
        output_tensors, self._feature_columns)
    return array_ops.concat(output_tensors, -1)

  def get_config(self):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    column_configs = serialization.serialize_feature_columns(
        self._feature_columns)
    config = {'feature_columns': column_configs}
    config['partitioner'] = generic_utils.serialize_keras_object(
        self._partitioner)

    base_config = super(  # pylint: disable=bad-super-call
        _BaseFeaturesLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    config_cp = config.copy()
    config_cp['feature_columns'] = serialization.deserialize_feature_columns(
        config['feature_columns'], custom_objects=custom_objects)
    config_cp['partitioner'] = generic_utils.deserialize_keras_object(
        config['partitioner'], custom_objects)

    return cls(**config_cp)
