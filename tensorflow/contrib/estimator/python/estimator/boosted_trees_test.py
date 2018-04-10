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
"""Tests boosted_trees estimators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.estimator.python.estimator import boosted_trees
from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.estimator.canned import boosted_trees as canned_boosted_trees
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_utils

NUM_FEATURES = 3

BUCKET_BOUNDARIES = [-2., .5, 12.]  # Boundaries for all the features.
INPUT_FEATURES = np.array(
    [
        [12.5, 1.0, -2.001, -2.0001, -1.999],  # feature_0 quantized:[3,2,0,0,1]
        [2.0, -3.0, 0.5, 0.0, 0.4995],         # feature_1 quantized:[2,0,2,1,1]
        [3.0, 20.0, 50.0, -100.0, 102.75],     # feature_2 quantized:[2,3,3,0,3]
    ],
    dtype=np.float32)
CLASSIFICATION_LABELS = [[0.], [1.], [1.], [0.], [0.]]
REGRESSION_LABELS = [[1.5], [0.3], [0.2], [2.], [5.]]
FEATURES_DICT = {'f_%d' % i: INPUT_FEATURES[i] for i in range(NUM_FEATURES)}


def _make_train_input_fn(is_classification):
  """Makes train input_fn for classification/regression."""

  def _input_fn():
    features = dict(FEATURES_DICT)
    if is_classification:
      labels = CLASSIFICATION_LABELS
    else:
      labels = REGRESSION_LABELS
    return features, labels

  return _input_fn


class BoostedTreesEstimatorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._head = canned_boosted_trees._create_regression_head(label_dimension=1)
    self._feature_columns = {
        feature_column.bucketized_column(
            feature_column.numeric_column('f_%d' % i, dtype=dtypes.float32),
            BUCKET_BOUNDARIES)
        for i in range(NUM_FEATURES)
    }

  def _assert_checkpoint(self, model_dir, global_step, finalized_trees,
                         attempted_layers):
    reader = checkpoint_utils.load_checkpoint(model_dir)
    self.assertEqual(global_step, reader.get_tensor(ops.GraphKeys.GLOBAL_STEP))
    serialized = reader.get_tensor('boosted_trees:0_serialized')
    ensemble_proto = boosted_trees_pb2.TreeEnsemble()
    ensemble_proto.ParseFromString(serialized)
    self.assertEqual(
        finalized_trees,
        sum([1 for t in ensemble_proto.tree_metadata if t.is_finalized]))
    self.assertEqual(attempted_layers,
                     ensemble_proto.growing_metadata.num_layers_attempted)

  def testTrainAndEvaluateEstimator(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees._BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        head=self._head,
        max_depth=5)

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 1.008551)

  def testInferEstimator(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees._BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        head=self._head)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    # Validate predictions.
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  def testBinaryClassifierTrainInMemoryAndEvalAndInfer(self):
    train_input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.boosted_trees_classifier_train_in_memory(
        train_input_fn=train_input_fn,
        feature_columns=self._feature_columns,
        n_trees=1,
        max_depth=5)
    # It will stop after 5 steps because of the max depth and num trees.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)

    # Check eval.
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    # Validate predictions.
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testRegressorTrainInMemoryAndEvalAndInfer(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.boosted_trees_regressor_train_in_memory(
        train_input_fn=train_input_fn,
        feature_columns=self._feature_columns,
        n_trees=1,
        max_depth=5)
    # It will stop after 5 steps because of the max depth and num trees.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)

    # Check eval.
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    # Validate predictions.
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])


if __name__ == '__main__':
  googletest.main()
