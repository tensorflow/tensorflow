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
"""This API defines FeatureColumn for sequential input.

NOTE: This API is a work in progress and will likely be changing frequently.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.feature_column import base_feature_layer as kfc
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.util.tf_export import keras_export

# pylint: disable=protected-access


@keras_export('keras.experimental.SequenceFeatures')
class SequenceFeatures(kfc._BaseFeaturesLayer):
  """A layer for sequence input.

    All `feature_columns` must be sequence dense columns with the same
    `sequence_length`. The output of this method can be fed into sequence
    networks, such as RNN.

    The output of this method is a 3D `Tensor` of shape `[batch_size, T, D]`.
    `T` is the maximum sequence length for this batch, which could differ from
    batch to batch.

    If multiple `feature_columns` are given with `Di` `num_elements` each, their
    outputs are concatenated. So, the final `Tensor` has shape
    `[batch_size, T, D0 + D1 + ... + Dn]`.

    Example:

    ```python
    # Behavior of some cells or feature columns may depend on whether we are in
    # training or inference mode, e.g. applying dropout.
    training = True
    rating = sequence_numeric_column('rating')
    watches = sequence_categorical_column_with_identity(
        'watches', num_buckets=1000)
    watches_embedding = embedding_column(watches, dimension=10)
    columns = [rating, watches_embedding]

    sequence_input_layer = SequenceFeatures(columns)
    features = tf.io.parse_example(...,
                                   features=make_parse_example_spec(columns))
    sequence_input, sequence_length = sequence_input_layer(
       features, training=training)
    sequence_length_mask = tf.sequence_mask(sequence_length)

    rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size, training=training)
    rnn_layer = tf.keras.layers.RNN(rnn_cell, training=training)
    outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
    ```
  """

  def __init__(
      self,
      feature_columns,
      trainable=True,
      name=None,
      **kwargs):
    """"Constructs a SequenceFeatures layer.

    Args:
      feature_columns: An iterable of dense sequence columns. Valid columns are
        - `embedding_column` that wraps a `sequence_categorical_column_with_*`
        - `sequence_numeric_column`.
      trainable: Boolean, whether the layer's variables will be updated via
        gradient descent during training.
      name: Name to give to the SequenceFeatures.
      **kwargs: Keyword arguments to construct a layer.

    Raises:
      ValueError: If any of the `feature_columns` is not a
        `SequenceDenseColumn`.
    """
    super(SequenceFeatures, self).__init__(
        feature_columns=feature_columns,
        trainable=trainable,
        name=name,
        expected_column_type=fc.SequenceDenseColumn,
        **kwargs)

  @property
  def _is_feature_layer(self):
    return True

  def _target_shape(self, input_shape, total_elements):
    return (input_shape[0], input_shape[1], total_elements)

  def call(self, features, training=None):
    """Returns sequence input corresponding to the `feature_columns`.

    Args:
      features: A dict mapping keys to tensors.
      training: Python boolean or None, indicating whether to the layer is being
        run in training mode. This argument is passed to the call method of any
        `FeatureColumn` that takes a `training` argument. For example, if a
        `FeatureColumn` performed dropout, the column could expose a `training`
        argument to control whether the dropout should be applied. If `None`,
        defaults to `tf.keras.backend.learning_phase()`.


    Returns:
      An `(input_layer, sequence_length)` tuple where:
      - input_layer: A float `Tensor` of shape `[batch_size, T, D]`.
          `T` is the maximum sequence length for this batch, which could differ
          from batch to batch. `D` is the sum of `num_elements` for all
          `feature_columns`.
      - sequence_length: An int `Tensor` of shape `[batch_size]`. The sequence
          length for each example.

    Raises:
      ValueError: If features are not a dictionary.
    """
    if not isinstance(features, dict):
      raise ValueError('We expected a dictionary here. Instead we got: ',
                       features)
    if training is None:
      training = backend.learning_phase()
    transformation_cache = fc.FeatureTransformationCache(features)
    output_tensors = []
    sequence_lengths = []

    for column in self._feature_columns:
      with ops.name_scope(column.name):
        try:
          dense_tensor, sequence_length = column.get_sequence_dense_tensor(
              transformation_cache, self._state_manager, training=training)
        except TypeError:
          dense_tensor, sequence_length = column.get_sequence_dense_tensor(
              transformation_cache, self._state_manager)
        # Flattens the final dimension to produce a 3D Tensor.
        output_tensors.append(self._process_dense_tensor(column, dense_tensor))
        sequence_lengths.append(sequence_length)

    # Check and process sequence lengths.
    fc._verify_static_batch_size_equality(sequence_lengths,
                                          self._feature_columns)
    sequence_length = _assert_all_equal_and_return(sequence_lengths)

    return self._verify_and_concat_tensors(output_tensors), sequence_length


def _assert_all_equal_and_return(tensors, name=None):
  """Asserts that all tensors are equal and returns the first one."""
  with ops.name_scope(name, 'assert_all_equal', values=tensors):
    if len(tensors) == 1:
      return tensors[0]
    assert_equal_ops = []
    for t in tensors[1:]:
      assert_equal_ops.append(check_ops.assert_equal(tensors[0], t))
    with ops.control_dependencies(assert_equal_ops):
      return array_ops.identity(tensors[0])
