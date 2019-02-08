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
"""Tests for GBDT estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import numpy as np

from tensorflow.contrib.boosted_trees.estimator_batch import estimator
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.layers.python.layers import feature_column as contrib_feature_column
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib as core_feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_utils


def _train_input_fn():
  features = {"x": constant_op.constant([[2.], [1.], [1.]])}
  label = constant_op.constant([[1], [0], [0]], dtype=dtypes.int32)
  return features, label


def _multiclass_train_input_fn():
  features = {
      "x": constant_op.constant([[2.], [1.], [1.], [5.], [3.5], [4.6], [3.5]])
  }
  label = constant_op.constant([[1], [0], [0], [2], [2], [0], [1]],
                               dtype=dtypes.int32)
  return features, label


def _ranking_train_input_fn():
  features = {
      "a.f1": constant_op.constant([[3.], [0.3], [1.]]),
      "a.f2": constant_op.constant([[0.1], [3.], [1.]]),
      "b.f1": constant_op.constant([[13.], [0.4], [5.]]),
      "b.f2": constant_op.constant([[1.], [3.], [0.01]]),
  }
  label = constant_op.constant([[0], [0], [1]], dtype=dtypes.int32)
  return features, label


def _eval_input_fn():
  features = {"x": constant_op.constant([[1.], [2.], [2.]])}
  label = constant_op.constant([[0], [1], [1]], dtype=dtypes.int32)
  return features, label


def _infer_ranking_train_input_fn():
  features = {
      "f1": constant_op.constant([[3.], [2], [1.]]),
      "f2": constant_op.constant([[0.1], [3.], [1.]])
  }
  return features, None


_QUANTILE_REGRESSION_SIZE = 1000


def _quantile_regression_input_fns(two_dimension=False):
  # The data generation is taken from
  # http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
  np.random.seed(1)

  def f(x):
    """The function to predict."""
    return x * np.sin(x)

  def g(x):
    """The function to predict."""
    return x * np.cos(x)

  #  Training data.
  x = np.atleast_2d(np.random.uniform(0, 10.0,
                                      size=_QUANTILE_REGRESSION_SIZE)).T
  x = x.astype(np.float32)

  # Labels.
  if not two_dimension:
    y = f(x).ravel()
  else:
    y = np.column_stack((f(x).ravel(), g(x).ravel()))

  # Add random noise.
  dy = 1.5 + 1.0 * np.random.random(y.shape)
  noise = np.random.normal(0, dy)
  y += noise
  y_original = y.astype(np.float32)
  if not two_dimension:
    y = y.reshape(_QUANTILE_REGRESSION_SIZE, 1)

  train_input_fn = numpy_io.numpy_input_fn(
      x=x,
      y=y,
      batch_size=_QUANTILE_REGRESSION_SIZE,
      num_epochs=None,
      shuffle=True)

  # Test on the training data to make sure the predictions are calibrated.
  test_input_fn = numpy_io.numpy_input_fn(
      x=x,
      y=y,
      batch_size=_QUANTILE_REGRESSION_SIZE,
      num_epochs=1,
      shuffle=False)

  return train_input_fn, test_input_fn, y_original


class BoostedTreeEstimatorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(self._export_dir_base)

  def _assert_checkpoint(self, model_dir, global_step):
    reader = checkpoint_utils.load_checkpoint(model_dir)
    self.assertEqual(global_step, reader.get_tensor(ops.GraphKeys.GLOBAL_STEP))

  def testFitAndEvaluateDontThrowException(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[contrib_feature_column.real_valued_column("x")])

    classifier.fit(input_fn=_train_input_fn, steps=15)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)
    classifier.export(self._export_dir_base)

  def testThatLeafIndexIsInPredictions(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[contrib_feature_column.real_valued_column("x")],
        output_leaf_index=True)

    classifier.fit(input_fn=_train_input_fn, steps=15)
    result_iter = classifier.predict(input_fn=_eval_input_fn)
    for prediction_dict in result_iter:
      self.assertTrue("leaf_index" in prediction_dict)
      self.assertTrue("logits" in prediction_dict)

  def testFitAndEvaluateDontThrowExceptionWithCoreForEstimator(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    # Use core head
    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    model = estimator.GradientBoostedDecisionTreeEstimator(
        head=head_fn,
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")],
        use_core_libs=True)

    model.fit(input_fn=_train_input_fn, steps=15)
    model.evaluate(input_fn=_eval_input_fn, steps=1)
    model.export(self._export_dir_base)

  def testFitAndEvaluateDontThrowExceptionWithCoreForClassifier(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")],
        use_core_libs=True)

    classifier.fit(input_fn=_train_input_fn, steps=15)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)
    classifier.export(self._export_dir_base)

  def testFitAndEvaluateDontThrowExceptionWithCoreForRegressor(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    regressor = estimator.GradientBoostedDecisionTreeRegressor(
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")],
        use_core_libs=True)

    regressor.fit(input_fn=_train_input_fn, steps=15)
    regressor.evaluate(input_fn=_eval_input_fn, steps=1)
    regressor.export(self._export_dir_base)

  def testRankingDontThrowExceptionForForEstimator(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    model = estimator.GradientBoostedDecisionTreeRanker(
        head=head_fn,
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        use_core_libs=True,
        feature_columns=[
            core_feature_column.numeric_column("f1"),
            core_feature_column.numeric_column("f2")
        ],
        ranking_model_pair_keys=("a", "b"))

    model.fit(input_fn=_ranking_train_input_fn, steps=1000)
    model.evaluate(input_fn=_ranking_train_input_fn, steps=1)
    model.predict(input_fn=_infer_ranking_train_input_fn)

  def testDoesNotOverrideGlobalSteps(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 2
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[contrib_feature_column.real_valued_column("x")],
        output_leaf_index=False)

    classifier.fit(input_fn=_train_input_fn, steps=15)
    # When no override of global steps, 5 steps were used.
    self._assert_checkpoint(classifier.model_dir, global_step=5)

  def testOverridesGlobalSteps(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 2
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[contrib_feature_column.real_valued_column("x")],
        output_leaf_index=False,
        override_global_step_value=10000000)

    classifier.fit(input_fn=_train_input_fn, steps=15)
    self._assert_checkpoint(classifier.model_dir, global_step=10000000)

  def testFitAndEvaluateMultiClassTreePerClassDontThrowException(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 3
    learner_config.constraints.max_tree_depth = 1
    learner_config.multi_class_strategy = (
        learner_pb2.LearnerConfig.TREE_PER_CLASS)

    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        n_classes=learner_config.num_classes,
        num_trees=1,
        examples_per_layer=7,
        model_dir=model_dir,
        config=config,
        feature_columns=[contrib_feature_column.real_valued_column("x")])

    classifier.fit(input_fn=_multiclass_train_input_fn, steps=100)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)
    classifier.export(self._export_dir_base)
    result_iter = classifier.predict(input_fn=_eval_input_fn)
    for prediction_dict in result_iter:
      self.assertTrue("classes" in prediction_dict)

  def testFitAndEvaluateMultiClassDiagonalDontThrowException(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 3
    learner_config.constraints.max_tree_depth = 1
    learner_config.multi_class_strategy = (
        learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)

    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        n_classes=learner_config.num_classes,
        num_trees=1,
        examples_per_layer=7,
        model_dir=model_dir,
        config=config,
        center_bias=False,
        feature_columns=[contrib_feature_column.real_valued_column("x")])

    classifier.fit(input_fn=_multiclass_train_input_fn, steps=100)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)
    classifier.export(self._export_dir_base)
    result_iter = classifier.predict(input_fn=_eval_input_fn)
    for prediction_dict in result_iter:
      self.assertTrue("classes" in prediction_dict)

  def testFitAndEvaluateMultiClassFullDontThrowException(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 3
    learner_config.constraints.max_tree_depth = 1
    learner_config.multi_class_strategy = (
        learner_pb2.LearnerConfig.FULL_HESSIAN)

    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.GradientBoostedDecisionTreeClassifier(
        learner_config=learner_config,
        n_classes=learner_config.num_classes,
        num_trees=1,
        examples_per_layer=7,
        model_dir=model_dir,
        config=config,
        center_bias=False,
        feature_columns=[contrib_feature_column.real_valued_column("x")])

    classifier.fit(input_fn=_multiclass_train_input_fn, steps=100)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)
    classifier.export(self._export_dir_base)
    result_iter = classifier.predict(input_fn=_eval_input_fn)
    for prediction_dict in result_iter:
      self.assertTrue("classes" in prediction_dict)

  # One dimensional quantile regression.
  def testQuantileRegression(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    learner_config.growing_mode = learner_pb2.LearnerConfig.WHOLE_TREE
    learner_config.constraints.min_node_weight = 1 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l2 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l1 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.tree_complexity = (
        1.0 / _QUANTILE_REGRESSION_SIZE)

    train_input_fn, test_input_fn, y = _quantile_regression_input_fns()

    # 95% percentile.
    model_upper = estimator.GradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.95],
        learner_config=learner_config,
        num_trees=100,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_upper.fit(input_fn=train_input_fn, steps=1000)
    result_iter = model_upper.predict(input_fn=test_input_fn)
    upper = []
    for prediction_dict in result_iter:
      upper.append(prediction_dict["scores"])

    frac_below_upper = round(1. * np.count_nonzero(upper > y) / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_below_upper >= 0.92)
    self.assertTrue(frac_below_upper <= 0.98)

    train_input_fn, test_input_fn, _ = _quantile_regression_input_fns()
    model_lower = estimator.GradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.05],
        learner_config=learner_config,
        num_trees=100,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_lower.fit(input_fn=train_input_fn, steps=1000)
    result_iter = model_lower.predict(input_fn=test_input_fn)
    lower = []
    for prediction_dict in result_iter:
      lower.append(prediction_dict["scores"])

    frac_above_lower = round(1. * np.count_nonzero(lower < y) / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_above_lower >= 0.92)
    self.assertTrue(frac_above_lower <= 0.98)

  # Multi-dimensional quantile regression.
  def testQuantileRegressionMultiDimLabel(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    learner_config.growing_mode = learner_pb2.LearnerConfig.WHOLE_TREE
    learner_config.constraints.min_node_weight = 1 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l2 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l1 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.tree_complexity = (
        1.0 / _QUANTILE_REGRESSION_SIZE)

    train_input_fn, test_input_fn, y = _quantile_regression_input_fns(
        two_dimension=True)

    # 95% percentile.
    model_upper = estimator.GradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.95],
        learner_config=learner_config,
        label_dimension=2,
        num_trees=100,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_upper.fit(input_fn=train_input_fn, steps=1000)
    result_iter = model_upper.predict(input_fn=test_input_fn)
    upper = []
    for prediction_dict in result_iter:
      upper.append(prediction_dict["scores"])

    count_below_upper = np.count_nonzero(upper > y, axis=0)
    count_both_below_upper = np.count_nonzero(np.prod(upper > y, axis=1))
    frac_below_upper_0 = round(1. * count_below_upper[0] / len(y), 3)
    frac_below_upper_1 = round(1. * count_below_upper[1] / len(y), 3)
    frac_both_below_upper = round(1. * count_both_below_upper / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_below_upper_0 >= 0.92)
    self.assertTrue(frac_below_upper_0 <= 0.98)
    self.assertTrue(frac_below_upper_1 >= 0.92)
    self.assertTrue(frac_below_upper_1 <= 0.98)
    self.assertTrue(frac_both_below_upper >= 0.91)
    self.assertTrue(frac_both_below_upper <= 0.99)

    train_input_fn, test_input_fn, _ = _quantile_regression_input_fns(
        two_dimension=True)
    model_lower = estimator.GradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.05],
        learner_config=learner_config,
        label_dimension=2,
        num_trees=100,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_lower.fit(input_fn=train_input_fn, steps=1000)
    result_iter = model_lower.predict(input_fn=test_input_fn)
    lower = []
    for prediction_dict in result_iter:
      lower.append(prediction_dict["scores"])

    count_above_lower = np.count_nonzero(lower < y, axis=0)
    count_both_aboce_lower = np.count_nonzero(np.prod(lower < y, axis=1))
    frac_above_lower_0 = round(1. * count_above_lower[0] / len(y), 3)
    frac_above_lower_1 = round(1. * count_above_lower[1] / len(y), 3)
    frac_both_above_lower = round(1. * count_both_aboce_lower / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_above_lower_0 >= 0.92)
    self.assertTrue(frac_above_lower_0 <= 0.98)
    self.assertTrue(frac_above_lower_1 >= 0.92)
    self.assertTrue(frac_above_lower_1 <= 0.98)
    self.assertTrue(frac_both_above_lower >= 0.91)
    self.assertTrue(frac_both_above_lower <= 0.99)


class CoreGradientBoostedDecisionTreeEstimators(test_util.TensorFlowTestCase):

  def testTrainEvaluateInferDoesNotThrowError(self):
    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    est = estimator.CoreGradientBoostedDecisionTreeEstimator(
        head=head_fn,
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")])

    # Train for a few steps.
    est.train(input_fn=_train_input_fn, steps=1000)
    est.evaluate(input_fn=_eval_input_fn, steps=1)
    est.predict(input_fn=_eval_input_fn)

  def testRankingDontThrowExceptionForForEstimator(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    est = estimator.CoreGradientBoostedDecisionTreeRanker(
        head=head_fn,
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=[
            core_feature_column.numeric_column("f1"),
            core_feature_column.numeric_column("f2")
        ],
        ranking_model_pair_keys=("a", "b"))

    # Train for a few steps.
    est.train(input_fn=_ranking_train_input_fn, steps=1000)
    est.evaluate(input_fn=_ranking_train_input_fn, steps=1)
    est.predict(input_fn=_infer_ranking_train_input_fn)

  def testFitAndEvaluateMultiClassTreePerClasssDontThrowException(self):
    n_classes = 3
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = n_classes
    learner_config.constraints.max_tree_depth = 1
    learner_config.multi_class_strategy = (
        learner_pb2.LearnerConfig.TREE_PER_CLASS)

    head_fn = estimator.core_multiclass_head(n_classes=n_classes)

    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.CoreGradientBoostedDecisionTreeEstimator(
        learner_config=learner_config,
        head=head_fn,
        num_trees=1,
        center_bias=False,
        examples_per_layer=7,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")])

    classifier.train(input_fn=_multiclass_train_input_fn, steps=100)
    classifier.evaluate(input_fn=_multiclass_train_input_fn, steps=1)
    classifier.predict(input_fn=_eval_input_fn)

  def testFitAndEvaluateMultiClassDiagonalDontThrowException(self):
    n_classes = 3
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = n_classes
    learner_config.constraints.max_tree_depth = 1
    learner_config.multi_class_strategy = (
        learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)

    head_fn = estimator.core_multiclass_head(n_classes=n_classes)

    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.CoreGradientBoostedDecisionTreeEstimator(
        learner_config=learner_config,
        head=head_fn,
        num_trees=1,
        center_bias=False,
        examples_per_layer=7,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")])

    classifier.train(input_fn=_multiclass_train_input_fn, steps=100)
    classifier.evaluate(input_fn=_multiclass_train_input_fn, steps=1)
    classifier.predict(input_fn=_eval_input_fn)

  def testFitAndEvaluateMultiClassFullDontThrowException(self):
    n_classes = 3
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = n_classes
    learner_config.constraints.max_tree_depth = 1
    learner_config.multi_class_strategy = (
        learner_pb2.LearnerConfig.FULL_HESSIAN)

    head_fn = estimator.core_multiclass_head(n_classes=n_classes)

    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.CoreGradientBoostedDecisionTreeEstimator(
        learner_config=learner_config,
        head=head_fn,
        num_trees=1,
        center_bias=False,
        examples_per_layer=7,
        model_dir=model_dir,
        config=config,
        feature_columns=[core_feature_column.numeric_column("x")])

    classifier.train(input_fn=_multiclass_train_input_fn, steps=100)
    classifier.evaluate(input_fn=_multiclass_train_input_fn, steps=1)
    classifier.predict(input_fn=_eval_input_fn)

  def testWeightedCategoricalColumn(self):
    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    feature_columns = [
        core_feature_column.weighted_categorical_column(
            categorical_column=core_feature_column
            .categorical_column_with_vocabulary_list(
                key="word", vocabulary_list=["the", "cat", "dog"]),
            weight_feature_key="weight")
    ]

    labels = np.array([[1], [1], [0], [0.]], dtype=np.float32)

    def _make_input_fn():

      def _input_fn():
        features_dict = {}
        # Sparse tensor representing
        # example 0: "cat","the"
        # examaple 1: "dog"
        # example 2: -
        # example 3: "the"
        # Weights for the words are 5 - cat, 6- dog and 1 -the.
        features_dict["word"] = sparse_tensor.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 0], [3, 0]],
            values=constant_op.constant(["the", "cat", "dog", "the"],
                                        dtype=dtypes.string),
            dense_shape=[4, 3])
        features_dict["weight"] = sparse_tensor.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 0], [3, 0]],
            values=[1., 5., 6., 1.],
            dense_shape=[4, 3])
        return features_dict, labels

      return _input_fn

    est = estimator.CoreGradientBoostedDecisionTreeEstimator(
        head=head_fn,
        learner_config=learner_config,
        num_trees=1,
        examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        feature_columns=feature_columns)

    input_fn = _make_input_fn()
    est.train(input_fn=input_fn, steps=100)
    est.evaluate(input_fn=input_fn, steps=1)
    est.predict(input_fn=input_fn)

  # One dimensional quantile regression.
  def testQuantileRegression(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    learner_config.growing_mode = learner_pb2.LearnerConfig.WHOLE_TREE
    learner_config.constraints.min_node_weight = 1 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l2 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l1 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.tree_complexity = (
        1.0 / _QUANTILE_REGRESSION_SIZE)

    train_input_fn, test_input_fn, y = _quantile_regression_input_fns()
    y = y.reshape(_QUANTILE_REGRESSION_SIZE, 1)

    # 95% percentile.
    model_upper = estimator.CoreGradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.95],
        learner_config=learner_config,
        num_trees=100,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_upper.train(input_fn=train_input_fn, steps=1000)
    result_iter = model_upper.predict(input_fn=test_input_fn)
    upper = []
    for prediction_dict in result_iter:
      upper.append(prediction_dict["predictions"])

    frac_below_upper = round(1. * np.count_nonzero(upper > y) / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_below_upper >= 0.92)
    self.assertTrue(frac_below_upper <= 0.98)

    train_input_fn, test_input_fn, _ = _quantile_regression_input_fns()
    model_lower = estimator.CoreGradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.05],
        learner_config=learner_config,
        num_trees=100,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_lower.train(input_fn=train_input_fn, steps=1000)
    result_iter = model_lower.predict(input_fn=test_input_fn)
    lower = []
    for prediction_dict in result_iter:
      lower.append(prediction_dict["predictions"])

    frac_above_lower = round(1. * np.count_nonzero(lower < y) / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_above_lower >= 0.92)
    self.assertTrue(frac_above_lower <= 0.98)

  # Multi-dimensional quantile regression.
  def testQuantileRegressionMultiDimLabel(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    learner_config.growing_mode = learner_pb2.LearnerConfig.WHOLE_TREE
    learner_config.constraints.min_node_weight = 1 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l2 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.l1 = 1.0 / _QUANTILE_REGRESSION_SIZE
    learner_config.regularization.tree_complexity = (
        1.0 / _QUANTILE_REGRESSION_SIZE)

    train_input_fn, test_input_fn, y = _quantile_regression_input_fns(
        two_dimension=True)
    y = y.reshape(_QUANTILE_REGRESSION_SIZE, 2)

    # 95% percentile.
    model_upper = estimator.CoreGradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.95],
        learner_config=learner_config,
        num_trees=100,
        label_dimension=2,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_upper.train(input_fn=train_input_fn, steps=1000)
    result_iter = model_upper.predict(input_fn=test_input_fn)
    upper = []
    for prediction_dict in result_iter:
      upper.append(prediction_dict["predictions"])

    count_below_upper = np.count_nonzero(upper > y, axis=0)
    count_both_below_upper = np.count_nonzero(np.prod(upper > y, axis=1))
    frac_below_upper_0 = round(1. * count_below_upper[0] / len(y), 3)
    frac_below_upper_1 = round(1. * count_below_upper[1] / len(y), 3)
    frac_both_below_upper = round(1. * count_both_below_upper / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_below_upper_0 >= 0.92)
    self.assertTrue(frac_below_upper_0 <= 0.98)
    self.assertTrue(frac_below_upper_1 >= 0.92)
    self.assertTrue(frac_below_upper_1 <= 0.98)
    self.assertTrue(frac_both_below_upper >= 0.91)
    self.assertTrue(frac_both_below_upper <= 0.99)

    train_input_fn, test_input_fn, _ = _quantile_regression_input_fns(
        two_dimension=True)
    model_lower = estimator.CoreGradientBoostedDecisionTreeQuantileRegressor(
        quantiles=[0.05],
        learner_config=learner_config,
        num_trees=100,
        label_dimension=2,
        examples_per_layer=_QUANTILE_REGRESSION_SIZE,
        center_bias=False)

    model_lower.train(input_fn=train_input_fn, steps=1000)
    result_iter = model_lower.predict(input_fn=test_input_fn)
    lower = []
    for prediction_dict in result_iter:
      lower.append(prediction_dict["predictions"])

    count_above_lower = np.count_nonzero(lower < y, axis=0)
    count_both_aboce_lower = np.count_nonzero(np.prod(lower < y, axis=1))
    frac_above_lower_0 = round(1. * count_above_lower[0] / len(y), 3)
    frac_above_lower_1 = round(1. * count_above_lower[1] / len(y), 3)
    frac_both_above_lower = round(1. * count_both_aboce_lower / len(y), 3)
    # +/- 3%
    self.assertTrue(frac_above_lower_0 >= 0.92)
    self.assertTrue(frac_above_lower_0 <= 0.98)
    self.assertTrue(frac_above_lower_1 >= 0.92)
    self.assertTrue(frac_above_lower_1 <= 0.98)
    self.assertTrue(frac_both_above_lower >= 0.91)
    self.assertTrue(frac_both_above_lower <= 0.99)


if __name__ == "__main__":
  googletest.main()
