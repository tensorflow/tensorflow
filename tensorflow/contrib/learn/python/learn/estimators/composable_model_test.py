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

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import composable_model
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test


def _iris_input_fn():
  iris = base.load_iris()
  return {
      'feature': constant_op.constant(
          iris.data, dtype=dtypes.float32)
  }, constant_op.constant(
      iris.target, shape=[150, 1], dtype=dtypes.int32)


def _base_model_fn(features, labels, mode, params):
  model = params['model']
  feature_columns = params['feature_columns']
  head = params['head']

  if mode == model_fn_lib.ModeKeys.TRAIN:
    logits = model.build_model(features, feature_columns, is_training=True)
  elif mode == model_fn_lib.ModeKeys.EVAL:
    logits = model.build_model(features, feature_columns, is_training=False)
  else:
    raise NotImplementedError

  def _train_op_fn(loss):
    global_step = contrib_variables.get_global_step()
    assert global_step
    train_step = model.get_train_step(loss)

    with ops.control_dependencies(train_step):
      with ops.get_default_graph().colocate_with(global_step):
        return state_ops.assign_add(global_step, 1).op

  return head.create_model_fn_ops(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)


def _linear_estimator(head, feature_columns):
  return estimator.Estimator(
      model_fn=_base_model_fn,
      params={
          'model':
              composable_model.LinearComposableModel(
                  num_label_columns=head.logits_dimension),
          'feature_columns':
              feature_columns,
          'head':
              head
      })


def _joint_linear_estimator(head, feature_columns):
  return estimator.Estimator(
      model_fn=_base_model_fn,
      params={
          'model':
              composable_model.LinearComposableModel(
                  num_label_columns=head.logits_dimension, _joint_weights=True),
          'feature_columns':
              feature_columns,
          'head':
              head
      })


def _dnn_estimator(head, feature_columns, hidden_units):
  return estimator.Estimator(
      model_fn=_base_model_fn,
      params={
          'model':
              composable_model.DNNComposableModel(
                  num_label_columns=head.logits_dimension,
                  hidden_units=hidden_units),
          'feature_columns':
              feature_columns,
          'head':
              head
      })


class ComposableModelTest(test.TestCase):

  def testLinearModel(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 100)
    age = feature_column.real_valued_column('age')

    head = head_lib._multi_class_head(n_classes=2)
    classifier = _linear_estimator(head, feature_columns=[age, language])

    classifier.fit(input_fn=input_fn, steps=1000)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=2000)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testJointLinearModel(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age':
              sparse_tensor.SparseTensor(
                  values=['1'], indices=[[0, 0]], dense_shape=[1, 1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 100)
    age = feature_column.sparse_column_with_hash_bucket('age', 2)

    head = head_lib._multi_class_head(n_classes=2)
    classifier = _joint_linear_estimator(head, feature_columns=[age, language])

    classifier.fit(input_fn=input_fn, steps=1000)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=2000)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testDNNModel(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]

    head = head_lib._multi_class_head(n_classes=3)
    classifier = _dnn_estimator(
        head, feature_columns=cont_features, hidden_units=[3, 3])

    classifier.fit(input_fn=_iris_input_fn, steps=1000)
    classifier.evaluate(input_fn=_iris_input_fn, steps=100)


if __name__ == '__main__':
  test.main()
