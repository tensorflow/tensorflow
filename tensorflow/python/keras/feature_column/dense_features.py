# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A layer that produces a dense `Tensor` based on given `feature_columns`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.feature_column import base_feature_layer as kfc
from tensorflow.python.util import serialization
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=['keras.layers.DenseFeatures'])
class DenseFeatures(kfc._BaseFeaturesLayer):  # pylint: disable=protected-access
  """A layer that produces a dense `Tensor` based on given `feature_columns`.

  Generally a single example in training data is described with FeatureColumns.
  At the first layer of the model, this column-oriented data should be converted
  to a single `Tensor`.

  This layer can be called multiple times with different features.

  This is the V1 version of this layer that uses variable_scope's or partitioner
  to create variables which works well with PartitionedVariables. Variable
  scopes are deprecated in V2, so the V2 version uses name_scopes instead. But
  currently that lacks support for partitioned variables. Use this if you need
  partitioned variables. Use the partitioner argument if you have a Keras model
  and uses `tf.compat.v1.keras.estimator.model_to_estimator` for training.

  Example:

  ```python
  price = tf.feature_column.numeric_column('price')
  keywords_embedded = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_hash_bucket("keywords", 10K),
      dimension=16)
  columns = [price, keywords_embedded, ...]
  partitioner = tf.compat.v1.fixed_size_partitioner(num_shards=4)
  feature_layer = tf.compat.v1.keras.layers.DenseFeatures(
      feature_columns=columns, partitioner=partitioner)

  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = feature_layer(features)
  for units in [128, 64, 32]:
    dense_tensor = tf.compat.v1.keras.layers.Dense(
                       units, activation='relu')(dense_tensor)
  prediction = tf.compat.v1.keras.layers.Dense(1)(dense_tensor)
  ```
  """

  def __init__(self,
               feature_columns,
               trainable=True,
               name=None,
               partitioner=None,
               **kwargs):
    """Constructs a DenseFeatures layer.

    Args:
      feature_columns: An iterable containing the FeatureColumns to use as
        inputs to your model. All items should be instances of classes derived
        from `DenseColumn` such as `numeric_column`, `embedding_column`,
        `bucketized_column`, `indicator_column`. If you have categorical
        features, you can wrap them with an `embedding_column` or
        `indicator_column`.
      trainable:  Boolean, whether the layer's variables will be updated via
        gradient descent during training.
      name: Name to give to the DenseFeatures.
      partitioner: Partitioner for input layer. Defaults to None.
      **kwargs: Keyword arguments to construct a layer.

    Raises:
      ValueError: if an item in `feature_columns` is not a `DenseColumn`.
    """
    super(DenseFeatures, self).__init__(
        feature_columns=feature_columns,
        trainable=trainable,
        name=name,
        partitioner=partitioner,
        expected_column_type=fc.DenseColumn,
        **kwargs)

  @property
  def _is_feature_layer(self):
    return True

  @property
  def _tracking_metadata(self):
    """String stored in metadata field in the SavedModel proto.

    Returns:
      A serialized JSON storing information necessary for recreating this layer.
    """
    metadata = json.loads(super(DenseFeatures, self)._tracking_metadata)
    metadata['_is_feature_layer'] = True
    return json.dumps(metadata, default=serialization.get_json_type)

  def _target_shape(self, input_shape, total_elements):
    return (input_shape[0], total_elements)

  def call(self, features, cols_to_output_tensors=None, training=None):
    """Returns a dense tensor corresponding to the `feature_columns`.

    Example usage:

    >>> t1 = tf.feature_column.embedding_column(
    ...    tf.feature_column.categorical_column_with_hash_bucket("t1", 2),
    ...    dimension=8)
    >>> t2 = tf.feature_column.numeric_column('t2')
    >>> feature_layer = tf.compat.v1.keras.layers.DenseFeatures([t1, t2])
    >>> features = {"t1": tf.constant(["a", "b"]), "t2": tf.constant([1, 2])}
    >>> dense_tensor = feature_layer(features, training=True)

    Args:
      features: A mapping from key to tensors. `FeatureColumn`s look up via
        these keys. For example `numeric_column('price')` will look at 'price'
        key in this dict. Values can be a `SparseTensor` or a `Tensor` depends
        on corresponding `FeatureColumn`.
      cols_to_output_tensors: If not `None`, this will be filled with a dict
        mapping feature columns to output tensors created.
      training: Python boolean or None, indicating whether to the layer is being
        run in training mode. This argument is passed to the call method of any
        `FeatureColumn` that takes a `training` argument. For example, if a
        `FeatureColumn` performed dropout, the column could expose a `training`
        argument to control whether the dropout should be applied. If `None`,
        defaults to `tf.keras.backend.learning_phase()`.


    Returns:
      A `Tensor` which represents input layer of a model. Its shape
      is (batch_size, first_layer_dimension) and its dtype is `float32`.
      first_layer_dimension is determined based on given `feature_columns`.

    Raises:
      ValueError: If features are not a dictionary.
    """
    if training is None:
      training = backend.learning_phase()
    if not isinstance(features, dict):
      raise ValueError('We expected a dictionary here. Instead we got: ',
                       features)
    transformation_cache = fc.FeatureTransformationCache(features)
    output_tensors = []
    for column in self._feature_columns:
      with ops.name_scope(column.name):
        try:
          tensor = column.get_dense_tensor(
              transformation_cache, self._state_manager, training=training)
        except TypeError:
          tensor = column.get_dense_tensor(transformation_cache,
                                           self._state_manager)
        processed_tensors = self._process_dense_tensor(column, tensor)
        if cols_to_output_tensors is not None:
          cols_to_output_tensors[column] = processed_tensors
        output_tensors.append(processed_tensors)
    return self._verify_and_concat_tensors(output_tensors)
