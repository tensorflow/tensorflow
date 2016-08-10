# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Support Vector Machine (SVM) Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer


class SVM(linear.LinearClassifier):
  """Support Vector Machine (SVM) model for binary classification.

  Currently, only linear SVMs are supported. For the underlying optimization
  problem, the SDCAOptimizer is used.

  Example Usage:
  ```
  real_feature_column = real_valued_column(...)
  sparse_feature_column = sparse_column_with_hash_bucket(...)

  estimator = SVM(
      example_id_column='example_id',
      feature_columns=[real_feature_column, sparse_feature_column],
      l2_regularization=10.0)

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...

  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features, otherwise there
  will be a `KeyError`:
    a feature with `key=example_id_column` whose value is a `Tensor` of dtype
    string.
    if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
      - if `feature_columns` is None, then `input` must contains only real
        valued `Tensor`.


  Parameters:
    example_id_column: A string defining the feature column name representing
      example ids. Used to initialize the underlying optimizer.
    feature_columns: An iterable containing all the feature columns used by the
      model. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    model_dir: Directory to save model parameters, graph and etc. This can also
        be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
    l1_regularization: L1-regularization parameter. Refers to global L1
    regularization (across all examples).
    l2_regularization: L2-regularization parameter. Refers to global L2
    regularization (across all examples).
    kernels: A list of kernels for the SVM. Currently, no kernels are supported.
      Reserved for future use for non-linear SVMs
    config: RunConfig object to configure the runtime settings.
  """

  def __init__(self,
               example_id_column,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               l1_regularization=0.0,
               l2_regularization=0.0,
               kernels=None,
               config=None):
    if kernels is not None:
      raise ValueError('Kernel SVMs are not currently supported.')
    optimizer = sdca_optimizer.SDCAOptimizer(
        example_id_column=example_id_column,
        symmetric_l1_regularization=l1_regularization,
        symmetric_l2_regularization=l2_regularization)

    super(SVM, self).__init__(
        model_dir=model_dir,
        n_classes=2,
        weight_column_name=weight_column_name,
        feature_columns=feature_columns,
        optimizer=optimizer,
        config=config)
    self._target_column = layers.binary_svm_target(
        weight_column_name=weight_column_name)

  def _loss_type(self):
    """Loss type used by SDCA Optimizer for linear SVM classification."""
    return 'hinge_loss'
