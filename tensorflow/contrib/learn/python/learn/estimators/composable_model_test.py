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

"""Tests for ComposableModel classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn.estimators import composable_model
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops


def _iris_input_fn():
  iris = tf.contrib.learn.datasets.load_iris()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=[150, 1], dtype=tf.int32)


class _BaseEstimatorForTest(estimator.BaseEstimator):

  def __init__(self,
               target_column,
               feature_columns):
    super(_BaseEstimatorForTest, self).__init__(model_dir=tempfile.mkdtemp())
    self._target_column = target_column
    self._feature_columns = feature_columns

  def _get_train_ops(self, features, targets):
    global_step = contrib_variables.get_global_step()
    assert global_step

    logits = self._model.build_model(
        features, self._feature_columns, is_training=True)
    loss = self._target_column.loss(logits, targets, features)
    train_step = self._model.get_train_step(loss)

    with ops.control_dependencies(train_step):
      with ops.get_default_graph().colocate_with(global_step):
        return state_ops.assign_add(global_step, 1).op, loss

  def _get_eval_ops(self, features, targets, metrics=None):
    logits = self._model.build_model(
        features, self._feature_columns, is_training=False)
    loss = self._target_column.loss(logits, targets, features)
    return {'loss': metrics_lib.streaming_mean(loss)}

  def _get_predict_ops(self, features):
    raise NotImplementedError


class LinearEstimator(_BaseEstimatorForTest):

  def __init__(self,
               target_column,
               feature_columns):
    super(LinearEstimator, self).__init__(target_column, feature_columns)
    self._model = composable_model.LinearComposableModel(
        num_label_columns=target_column.num_label_columns)


class DNNEstimator(_BaseEstimatorForTest):

  def __init__(self,
               target_column,
               feature_columns,
               hidden_units):
    super(DNNEstimator, self).__init__(target_column, feature_columns)
    self._model = composable_model.DNNComposableModel(
        num_label_columns=target_column.num_label_columns,
        hidden_units=hidden_units)


class ComposableModelTest(tf.test.TestCase):

  def testLinearModel(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age': tf.constant([1]),
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    age = tf.contrib.layers.real_valued_column('age')

    target_column = layers.multi_class_target(n_classes=2)
    classifier = LinearEstimator(target_column,
                                 feature_columns=[age, language])

    classifier.fit(input_fn=input_fn, steps=1000)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=2000)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testDNNModel(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    target_column = layers.multi_class_target(n_classes=3)
    classifier = DNNEstimator(target_column,
                              feature_columns=cont_features,
                              hidden_units=[3, 3])

    classifier.fit(input_fn=_iris_input_fn, steps=1000)
    classifier.evaluate(input_fn=_iris_input_fn, steps=100)


if __name__ == '__main__':
  tf.test.main()
