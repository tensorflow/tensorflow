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
"""Tests for TensorForestTrainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib as core_feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils


def _get_classification_input_fns():
  iris = base.load_iris()
  data = iris.data.astype(np.float32)
  labels = iris.target.astype(np.int32)

  train_input_fn = numpy_io.numpy_input_fn(
      x=data, y=labels, batch_size=150, num_epochs=None, shuffle=False)

  predict_input_fn = numpy_io.numpy_input_fn(
      x=data[:1,], y=None, batch_size=1, num_epochs=1, shuffle=False)
  return train_input_fn, predict_input_fn


def _get_regression_input_fns():
  boston = base.load_boston()
  data = boston.data.astype(np.float32)
  labels = boston.target.astype(np.int32)

  train_input_fn = numpy_io.numpy_input_fn(
      x=data, y=labels, batch_size=506, num_epochs=None, shuffle=False)

  predict_input_fn = numpy_io.numpy_input_fn(
      x=data[:1,], y=None, batch_size=1, num_epochs=1, shuffle=False)
  return train_input_fn, predict_input_fn


class TensorForestTrainerTests(test.TestCase):

  def testClassification(self):
    """Tests multi-class classification using matrix data as input."""
    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)
    classifier = random_forest.TensorForestEstimator(hparams.fill())

    input_fn, predict_input_fn = _get_classification_input_fns()
    classifier.fit(input_fn=input_fn, steps=100)
    res = classifier.evaluate(input_fn=input_fn, steps=10)

    self.assertEqual(1.0, res['accuracy'])
    self.assertAllClose(0.55144483, res['loss'])

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0.576117, 0.211942, 0.211942]],
                        [pred['probabilities'] for pred in predictions])

  def testRegression(self):
    """Tests regression using matrix data as input."""

    hparams = tensor_forest.ForestHParams(
        num_trees=5,
        max_nodes=1000,
        num_classes=1,
        num_features=13,
        regression=True,
        split_after_samples=20)

    regressor = random_forest.TensorForestEstimator(hparams.fill())

    input_fn, predict_input_fn = _get_regression_input_fns()

    regressor.fit(input_fn=input_fn, steps=100)
    res = regressor.evaluate(input_fn=input_fn, steps=10)
    self.assertGreaterEqual(0.1, res['loss'])

    predictions = list(regressor.predict(input_fn=predict_input_fn))
    self.assertAllClose([24.], [pred['scores'] for pred in predictions], atol=1)

  def testAdditionalOutputs(self):
    """Tests multi-class classification using matrix data as input."""
    hparams = tensor_forest.ForestHParams(
        num_trees=1,
        max_nodes=100,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)
    classifier = random_forest.TensorForestEstimator(
        hparams.fill(), keys_column='keys', include_all_in_serving=True)

    iris = base.load_iris()
    data = iris.data.astype(np.float32)
    labels = iris.target.astype(np.int32)

    input_fn = numpy_io.numpy_input_fn(
        x={
            'x': data,
            'keys': np.arange(len(iris.data)).reshape(150, 1)
        },
        y=labels,
        batch_size=10,
        num_epochs=1,
        shuffle=False)

    classifier.fit(input_fn=input_fn, steps=100)
    predictions = list(classifier.predict(input_fn=input_fn))
    # Check that there is a key column, tree paths and var.
    for pred in predictions:
      self.assertTrue('keys' in pred)
      self.assertTrue('tree_paths' in pred)
      self.assertTrue('prediction_variance' in pred)

  def _assert_checkpoint(self, model_dir, global_step):
    reader = checkpoint_utils.load_checkpoint(model_dir)
    self.assertLessEqual(
        reader.get_tensor(ops.GraphKeys.GLOBAL_STEP), global_step)

  def testEarlyStopping(self):
    """Tests multi-class classification using matrix data as input."""
    hparams = tensor_forest.ForestHParams(
        num_trees=100,
        max_nodes=10000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)
    classifier = random_forest.TensorForestEstimator(
        hparams.fill(),
        # Set a crazy threshold - 30% loss change.
        early_stopping_loss_threshold=0.3,
        early_stopping_rounds=2)

    input_fn, _ = _get_classification_input_fns()
    classifier.fit(input_fn=input_fn, steps=100)

    # We stopped early.
    self._assert_checkpoint(classifier.model_dir, global_step=5)


class CoreTensorForestTests(test.TestCase):

  def testTrainEvaluateInferDoesNotThrowErrorForClassifier(self):
    head_fn = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)

    est = random_forest.CoreTensorForestEstimator(hparams.fill(), head=head_fn)

    input_fn, predict_input_fn = _get_classification_input_fns()

    est.train(input_fn=input_fn, steps=100)
    res = est.evaluate(input_fn=input_fn, steps=1)

    self.assertEqual(1.0, res['accuracy'])
    self.assertAllClose(0.55144483, res['loss'])

    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0.576117, 0.211942, 0.211942]],
                        [pred['probabilities'] for pred in predictions])

  def testRegression(self):
    """Tests regression using matrix data as input."""
    head_fn = head_lib._regression_head(
        label_dimension=1,
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    hparams = tensor_forest.ForestHParams(
        num_trees=5,
        max_nodes=1000,
        num_classes=1,
        num_features=13,
        regression=True,
        split_after_samples=20)

    regressor = random_forest.CoreTensorForestEstimator(
        hparams.fill(), head=head_fn)

    input_fn, predict_input_fn = _get_regression_input_fns()

    regressor.train(input_fn=input_fn, steps=100)
    res = regressor.evaluate(input_fn=input_fn, steps=10)
    self.assertGreaterEqual(0.1, res['loss'])

    predictions = list(regressor.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[24.]], [pred['predictions'] for pred in predictions], atol=1)

  def testWithFeatureColumns(self):
    head_fn = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)

    est = random_forest.CoreTensorForestEstimator(
        hparams.fill(),
        head=head_fn,
        feature_columns=[core_feature_column.numeric_column('x')])

    iris = base.load_iris()
    data = {'x': iris.data.astype(np.float32)}
    labels = iris.target.astype(np.int32)

    input_fn = numpy_io.numpy_input_fn(
        x=data, y=labels, batch_size=150, num_epochs=None, shuffle=False)

    est.train(input_fn=input_fn, steps=100)
    res = est.evaluate(input_fn=input_fn, steps=1)

    self.assertEqual(1.0, res['accuracy'])
    self.assertAllClose(0.55144483, res['loss'])

  def testAutofillsClassificationHead(self):
    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)

    est = random_forest.CoreTensorForestEstimator(hparams.fill())

    input_fn, _ = _get_classification_input_fns()

    est.train(input_fn=input_fn, steps=100)
    res = est.evaluate(input_fn=input_fn, steps=1)

    self.assertEqual(1.0, res['accuracy'])
    self.assertAllClose(0.55144483, res['loss'])

  def testAutofillsRegressionHead(self):
    hparams = tensor_forest.ForestHParams(
        num_trees=5,
        max_nodes=1000,
        num_classes=1,
        num_features=13,
        regression=True,
        split_after_samples=20)

    regressor = random_forest.CoreTensorForestEstimator(hparams.fill())

    input_fn, predict_input_fn = _get_regression_input_fns()

    regressor.train(input_fn=input_fn, steps=100)
    res = regressor.evaluate(input_fn=input_fn, steps=10)
    self.assertGreaterEqual(0.1, res['loss'])

    predictions = list(regressor.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[24.]], [pred['predictions'] for pred in predictions], atol=1)

  def testAdditionalOutputs(self):
    """Tests multi-class classification using matrix data as input."""
    hparams = tensor_forest.ForestHParams(
        num_trees=1,
        max_nodes=100,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)
    classifier = random_forest.CoreTensorForestEstimator(
        hparams.fill(), keys_column='keys', include_all_in_serving=True)

    iris = base.load_iris()
    data = iris.data.astype(np.float32)
    labels = iris.target.astype(np.int32)

    input_fn = numpy_io.numpy_input_fn(
        x={
            'x': data,
            'keys': np.arange(len(iris.data)).reshape(150, 1)
        },
        y=labels,
        batch_size=10,
        num_epochs=1,
        shuffle=False)

    classifier.train(input_fn=input_fn, steps=100)
    predictions = list(classifier.predict(input_fn=input_fn))
    # Check that there is a key column, tree paths and var.
    for pred in predictions:
      self.assertTrue('keys' in pred)
      self.assertTrue('tree_paths' in pred)
      self.assertTrue('prediction_variance' in pred)

  def _assert_checkpoint(self, model_dir, global_step):
    reader = checkpoint_utils.load_checkpoint(model_dir)
    self.assertLessEqual(
        reader.get_tensor(ops.GraphKeys.GLOBAL_STEP), global_step)

  def testEarlyStopping(self):
    head_fn = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)

    est = random_forest.CoreTensorForestEstimator(
        hparams.fill(),
        head=head_fn,
        # Set a crazy threshold - 30% loss change.
        early_stopping_loss_threshold=0.3,
        early_stopping_rounds=2)

    input_fn, _ = _get_classification_input_fns()
    est.train(input_fn=input_fn, steps=100)
    # We stopped early.
    self._assert_checkpoint(est.model_dir, global_step=8)


if __name__ == "__main__":
  test.main()
