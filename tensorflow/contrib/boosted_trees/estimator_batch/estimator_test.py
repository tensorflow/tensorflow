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
from tensorflow.contrib.boosted_trees.estimator_batch import estimator
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.layers.python.layers import feature_column as contrib_feature_column
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.feature_column import feature_column_lib as core_feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


def _train_input_fn():
  features = {"x": constant_op.constant([[2.], [1.], [1.]])}
  label = constant_op.constant([[1], [0], [0]], dtype=dtypes.int32)
  return features, label


def _eval_input_fn():
  features = {"x": constant_op.constant([[1.], [2.], [2.]])}
  label = constant_op.constant([[0], [1], [1]], dtype=dtypes.int32)
  return features, label


class BoostedTreeEstimatorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(self._export_dir_base)

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


if __name__ == "__main__":
  googletest.main()
