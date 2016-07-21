#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Linear Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

from tensorflow.contrib import layers
from tensorflow.contrib.linear_optimizer.python.ops import sdca_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


class SDCAOptimizer(object):
  """Wrapper class for SDCA optimizer.

  Example usage:
    real_feature_column = real_valued_column(...)
    sparse_feature_column = sparse_column_with_hash_bucket(...)
    sdca_optimizer = linear.SDCAOptimizer(example_id_column='example_id',
                                          symmetric_l2_regularization=2.0)
    classifier = linear.LinearClassifier(
        feature_columns=[real_feature_column, sparse_feature_column],
        weight_column_name=...,
        optimizer=sdca_optimizer)
    classifier.train(input_fn_train, steps=50)
    classifier.evaluate(input_fn=input_fn_eval)

  Here the expectation is that the input_fn_* functions passed to train and
  evaluate return a pair of (dict, label_tensor) where dict has an "example_id"
  key whose value is a tensor of shape [batch_size] and dtype string.
  """

  def __init__(self,
               example_id_column,
               symmetric_l1_regularization=0.0,
               symmetric_l2_regularization=1.0):
    self._example_id_column = example_id_column
    self._symmetric_l1_regularization = symmetric_l1_regularization
    self._symmetric_l2_regularization = symmetric_l2_regularization

  def get_train_step(self, linear_feature_columns, weight_column_name,
                     loss_type, features, targets, columns_to_variables,
                     global_step):
    """Returns the training operation of an SDCAModel optimizer."""

    # TODO(sibyl-vie3Poto): Rename this method to convert_to_sparse_tensor and move under
    # contrib/framework.
    def _dense_to_sparse_tensor(dense_tensor):
      """Returns a SparseTensor for the input dense_tensor."""
      ignore_value = 0.0
      sparse_indices = array_ops.where(math_ops.not_equal(
          dense_tensor, math_ops.cast(ignore_value, dense_tensor.dtype)))
      sparse_values = array_ops.gather_nd(dense_tensor, sparse_indices)
      # SparseTensor needs the shape to be converted to int64.
      int64_shape = math_ops.to_int64(array_ops.shape(dense_tensor))
      return ops.SparseTensor(sparse_indices, sparse_values, shape=int64_shape)

    def _training_examples_and_variables():
      """Returns dictionaries for training examples and variables."""
      batch_size = targets.get_shape()[0]

      # Iterate over all feature columns and create appropriate lists for dense
      # and sparse features as well as dense and sparse weights (variables) for
      # SDCA.
      # TODO(sibyl-vie3Poto): Reshape variables stored as values in column_to_variables
      # dict as 1-dimensional tensors.
      dense_features, sparse_features = [], []
      dense_features_weights, sparse_features_weights = [], []
      for column in sorted(set(linear_feature_columns), key=lambda x: x.key):
        transformed_tensor = features[column]
        if isinstance(column, layers.feature_column.
                      _RealValuedColumn):  # pylint: disable=protected-access
          # A real-valued column corresponds to a dense feature in SDCA.
          if column.dimension != 1:
            raise ValueError(
                "Invalid column dimension %d for column %s. SDCAOptimizer "
                "supports only 1-dimensional dense feature columns." %
                (column.dimension, column.name))

          dense_features.append(array_ops.reshape(transformed_tensor,
                                                  shape=[-1]))
          # For real valued columns, the variables list contains exactly one
          # element.
          dense_features_weights.append(columns_to_variables[column][0])
        elif isinstance(column, layers.feature_column.
                        _BucketizedColumn):  # pylint: disable=protected-access
          # A bucketized column corresponds to a sparse feature in SDCA. The
          # bucketized feature is "sparsified" for SDCA by converting it to a
          # SparseTensor respresenting the one-hot encoding of the bucketized
          # feature.
          dense_bucket_tensor = column.to_dnn_input_layer(transformed_tensor)
          sparse_bucket_tensor = _dense_to_sparse_tensor(dense_bucket_tensor)
          sparse_features.append(sparse_bucket_tensor)
          # For bucketized columns, the variables list contains exactly one
          # element.
          sparse_features_weights.append(columns_to_variables[column][0])
        elif isinstance(column,
                        (layers.feature_column.
                         _CrossedColumn,  # pylint: disable=protected-access
                         layers.feature_column._SparseColumn
                        )):  # pylint: disable=protected-access
          weights_tensor = ops.SparseTensor(
              indices=transformed_tensor.indices,
              values=array_ops.ones_like(transformed_tensor.values),
              shape=transformed_tensor.shape)
          sparse_features_tensor = sparse_ops.sparse_merge(transformed_tensor,
                                                           weights_tensor,
                                                           column.length)
          sparse_features.append(math_ops.to_float(sparse_features_tensor))
          sparse_features_weights.append(columns_to_variables[column][0])
        elif isinstance(
            column,
            layers.feature_column._WeightedSparseColumn):  # pylint: disable=protected-access
          id_tensor = column.id_tensor(transformed_tensor)
          weight_tensor = column.weight_tensor(transformed_tensor)
          sparse_features_tensor = sparse_ops.sparse_merge(
              id_tensor, weight_tensor, column.length,
              name="{}_sparse_merge".format(column.name))
          sparse_features.append(math_ops.to_float(
              sparse_features_tensor, name="{}_to_float".format(column.name)))
          sparse_features_weights.append(columns_to_variables[column][0])
        else:
          raise ValueError("SDCAOptimizer does not support column type %s." %
                           type(column).__name__)

      example_weights = array_ops.reshape(
          features[weight_column_name],
          shape=[-1]) if weight_column_name else array_ops.ones([batch_size])
      example_ids = features[self._example_id_column]
      examples = dict(
          sparse_features=sparse_features,
          dense_features=dense_features,
          example_labels=math_ops.to_float(
              array_ops.reshape(targets, shape=[-1])),
          example_weights=example_weights,
          example_ids=example_ids)
      sdca_variables = dict(sparse_features_weights=sparse_features_weights,
                            dense_features_weights=dense_features_weights)
      return examples, sdca_variables

    options = dict(
        symmetric_l1_regularization=self._symmetric_l1_regularization,
        symmetric_l2_regularization=self._symmetric_l2_regularization,
        loss_type=loss_type)
    training_examples, training_variables = _training_examples_and_variables()
    # TODO(sibyl-vie3Poto): Take care of cleanup, when the API to reset the container is
    # available.
    sdca_model = sdca_ops.SdcaModel(container=uuid.uuid4().hex,
                                    examples=training_examples,
                                    variables=training_variables,
                                    options=options)
    return sdca_model.minimize(global_step=global_step)
