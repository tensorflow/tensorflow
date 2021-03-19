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

import collections
import re

from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import tensor_shape
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
    self._feature_columns = _normalize_feature_columns(
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
      with variable_scope.variable_scope(
          self.name, partitioner=self._partitioner):
        with variable_scope.variable_scope(
            _sanitize_column_name_for_variable_scope(column.name)):
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
    _verify_static_batch_size_equality(output_tensors, self._feature_columns)
    return array_ops.concat(output_tensors, -1)

  def get_config(self):
    # Import here to avoid circular imports.
    from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top
    column_configs = [serialization.serialize_feature_column(fc)
                      for fc in self._feature_columns]
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
    columns_by_name = {}
    config_cp['feature_columns'] = [serialization.deserialize_feature_column(
        c, custom_objects, columns_by_name) for c in config['feature_columns']]
    config_cp['partitioner'] = generic_utils.deserialize_keras_object(
        config['partitioner'], custom_objects)

    return cls(**config_cp)


def _sanitize_column_name_for_variable_scope(name):
  """Sanitizes user-provided feature names for use as variable scopes."""
  invalid_char = re.compile('[^A-Za-z0-9_.\\-]')
  return invalid_char.sub('_', name)


def _verify_static_batch_size_equality(tensors, columns):
  """Verify equality between static batch sizes.

  Args:
    tensors: iterable of input tensors.
    columns: Corresponding feature columns.

  Raises:
    ValueError: in case of mismatched batch sizes.
  """
  expected_batch_size = None
  for i in range(0, len(tensors)):
    # bath_size is a Dimension object.
    batch_size = tensor_shape.Dimension(tensor_shape.dimension_value(
        tensors[i].shape[0]))
    if batch_size.value is not None:
      if expected_batch_size is None:
        bath_size_column_index = i
        expected_batch_size = batch_size
      elif not expected_batch_size.is_compatible_with(batch_size):
        raise ValueError(
            'Batch size (first dimension) of each feature must be same. '
            'Batch size of columns ({}, {}): ({}, {})'.format(
                columns[bath_size_column_index].name, columns[i].name,
                expected_batch_size, batch_size))


def _normalize_feature_columns(feature_columns):
  """Normalizes the `feature_columns` input.

  This method converts the `feature_columns` to list type as best as it can. In
  addition, verifies the type and other parts of feature_columns, required by
  downstream library.

  Args:
    feature_columns: The raw feature columns, usually passed by users.

  Returns:
    The normalized feature column list.

  Raises:
    ValueError: for any invalid inputs, such as empty, duplicated names, etc.
  """
  if isinstance(feature_columns, feature_column_v2.FeatureColumn):
    feature_columns = [feature_columns]

  if isinstance(feature_columns, collections.Iterator):
    feature_columns = list(feature_columns)

  if isinstance(feature_columns, dict):
    raise ValueError('Expected feature_columns to be iterable, found dict.')

  for column in feature_columns:
    if not isinstance(column, feature_column_v2.FeatureColumn):
      raise ValueError('Items of feature_columns must be a FeatureColumn. '
                       'Given (type {}): {}.'.format(type(column), column))
  if not feature_columns:
    raise ValueError('feature_columns must not be empty.')
  name_to_column = {}
  for column in feature_columns:
    if column.name in name_to_column:
      raise ValueError('Duplicate feature column name found for columns: {} '
                       'and {}. This usually means that these columns refer to '
                       'same base feature. Either one must be discarded or a '
                       'duplicated but renamed item must be inserted in '
                       'features dict.'.format(column,
                                               name_to_column[column.name]))
    name_to_column[column.name] = column

  return sorted(feature_columns, key=lambda x: x.name)
