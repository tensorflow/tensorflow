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
"""Tests for combined DNN + GBDT estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from tensorflow.contrib.boosted_trees.estimator_batch import dnn_tree_combined_estimator as estimator
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.estimator import exporter
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.export import export
from tensorflow.python.ops import parsing_ops
from tensorflow.python.feature_column import feature_column_lib as core_feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_utils


def _train_input_fn():
  features = {
      "x": constant_op.constant([[2.], [1.], [1.]])
  }
  label = constant_op.constant([[1], [0], [0]], dtype=dtypes.int32)
  return features, label


def _eval_input_fn():
  features = {
      "x": constant_op.constant([[1.], [2.], [2.]])
  }
  label = constant_op.constant([[0], [1], [1]], dtype=dtypes.int32)
  return features, label


class DNNBoostedTreeCombinedTest(test_util.TensorFlowTestCase):

  def testClassifierContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, estimator.DNNBoostedTreeCombinedClassifier)

  def testRegressorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, estimator.DNNBoostedTreeCombinedRegressor)

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, estimator.DNNBoostedTreeCombinedEstimator)

  def testNoDNNFeatureColumns(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2

    with self.assertRaisesRegexp(
        ValueError,
        "dnn_feature_columns must be specified"):
      classifier = estimator.DNNBoostedTreeCombinedClassifier(
          dnn_hidden_units=[1],
          dnn_feature_columns=[],
          tree_learner_config=learner_config,
          num_trees=1,
          tree_examples_per_layer=3,
          n_classes=2)
      classifier.fit(input_fn=_train_input_fn, steps=5)

  def testFitAndEvaluateDontThrowException(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.DNNBoostedTreeCombinedClassifier(
        dnn_hidden_units=[1],
        dnn_feature_columns=[feature_column.real_valued_column("x")],
        tree_learner_config=learner_config,
        num_trees=1,
        tree_examples_per_layer=3,
        n_classes=2,
        model_dir=model_dir,
        config=config,
        dnn_steps_to_train=10,
        dnn_input_layer_to_tree=False,
        tree_feature_columns=[feature_column.real_valued_column("x")])

    classifier.fit(input_fn=_train_input_fn, steps=15)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)

  def testFitAndEvaluateWithDistillation(self):
    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 1
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    classifier = estimator.DNNBoostedTreeCombinedClassifier(
        dnn_hidden_units=[1],
        dnn_feature_columns=[feature_column.real_valued_column("x")],
        tree_learner_config=learner_config,
        num_trees=1,
        tree_examples_per_layer=3,
        n_classes=2,
        model_dir=model_dir,
        config=config,
        dnn_steps_to_train=10,
        dnn_input_layer_to_tree=False,
        tree_feature_columns=[feature_column.real_valued_column("x")],
        dnn_to_tree_distillation_param=(1, None))

    classifier.fit(input_fn=_train_input_fn, steps=15)
    classifier.evaluate(input_fn=_eval_input_fn, steps=1)


class CoreDNNBoostedTreeCombinedTest(test_util.TensorFlowTestCase):

  def _assert_checkpoint(self, model_dir, global_step):
    reader = checkpoint_utils.load_checkpoint(model_dir)
    self.assertEqual(global_step, reader.get_tensor(ops.GraphKeys.GLOBAL_STEP))

  def testTrainEvaluateInferDoesNotThrowErrorWithNoDnnInput(self):
    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    est = estimator.CoreDNNBoostedTreeCombinedEstimator(
        head=head_fn,
        dnn_hidden_units=[1],
        dnn_feature_columns=[core_feature_column.numeric_column("x")],
        tree_learner_config=learner_config,
        num_trees=1,
        tree_examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        dnn_steps_to_train=10,
        dnn_input_layer_to_tree=False,
        tree_feature_columns=[core_feature_column.numeric_column("x")])

    # Train for a few steps.
    est.train(input_fn=_train_input_fn, steps=1000)
    # 10 steps for dnn, 3  for 1 tree of depth 3 + 1 after the tree finished
    self._assert_checkpoint(est.model_dir, global_step=14)
    res = est.evaluate(input_fn=_eval_input_fn, steps=1)
    self.assertLess(0.5, res["auc"])
    est.predict(input_fn=_eval_input_fn)

  def testTrainEvaluateInferDoesNotThrowErrorWithDnnInput(self):
    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    est = estimator.CoreDNNBoostedTreeCombinedEstimator(
        head=head_fn,
        dnn_hidden_units=[1],
        dnn_feature_columns=[core_feature_column.numeric_column("x")],
        tree_learner_config=learner_config,
        num_trees=1,
        tree_examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        dnn_steps_to_train=10,
        dnn_input_layer_to_tree=True,
        tree_feature_columns=[])

    # Train for a few steps.
    est.train(input_fn=_train_input_fn, steps=1000)
    res = est.evaluate(input_fn=_eval_input_fn, steps=1)
    self.assertLess(0.5, res["auc"])
    est.predict(input_fn=_eval_input_fn)

  def testTrainEvaluateWithDnnForInputAndTreeForPredict(self):
    head_fn = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS)

    learner_config = learner_pb2.LearnerConfig()
    learner_config.num_classes = 2
    learner_config.constraints.max_tree_depth = 3
    model_dir = tempfile.mkdtemp()
    config = run_config.RunConfig()

    est = estimator.CoreDNNBoostedTreeCombinedEstimator(
        head=head_fn,
        dnn_hidden_units=[1],
        dnn_feature_columns=[core_feature_column.numeric_column("x")],
        tree_learner_config=learner_config,
        num_trees=1,
        tree_examples_per_layer=3,
        model_dir=model_dir,
        config=config,
        dnn_steps_to_train=10,
        dnn_input_layer_to_tree=True,
        predict_with_tree_only=True,
        dnn_to_tree_distillation_param=(0.5, None),
        tree_feature_columns=[])

    # Train for a few steps.
    est.train(input_fn=_train_input_fn, steps=1000)
    res = est.evaluate(input_fn=_eval_input_fn, steps=1)
    self.assertLess(0.5, res["auc"])
    est.predict(input_fn=_eval_input_fn)
    serving_input_fn = (
        export.build_parsing_serving_input_receiver_fn(
            feature_spec={"x": parsing_ops.FixedLenFeature(
                [1], dtype=dtypes.float32)}))
    base_exporter = exporter.FinalExporter(
        name="Servo",
        serving_input_receiver_fn=serving_input_fn,
        assets_extra=None)
    export_path = os.path.join(model_dir, "export")
    base_exporter.export(
        est,
        export_path=export_path,
        checkpoint_path=None,
        eval_result={},
        is_the_final_export=True)

if __name__ == "__main__":
  googletest.main()
