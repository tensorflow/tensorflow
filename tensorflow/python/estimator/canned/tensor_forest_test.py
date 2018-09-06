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
"""Tests tensor_forest estimators and model_fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.estimator.canned import tensor_forest
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_utils

NUM_FEATURES = 3
INPUT_FEATURES = np.array(
    [
        # feature_0
        [12.5, 1.0, -2.001, -2.0001, -1.999],
        # feature_1
        [2.0, -3.0, 0.5, 0.0, 0.4995],
        # feature_2
        [3.0, 20.0, 50.0, -100.0, 102.75],
    ],
    dtype=np.float32)

CLASSIFICATION_LABELS = [[0.], [1.], [1.], [0.], [0.]]
REGRESSION_LABELS = [[1.5], [0.3], [0.2], [2.], [5.]]
FEATURES_DICT = {'f_%d' % i: INPUT_FEATURES[i] for i in range(NUM_FEATURES)}


def _make_train_input_fn(is_classification):
  """Makes train input_fn for classification/regression."""

  def _input_fn():
    features_dict = dict(FEATURES_DICT)  # copies the dict to add an entry.
    features_dict[EXAMPLE_ID_COLUMN] = constant_op.constant(EXAMPLE_IDS)
    labels = CLASSIFICATION_LABELS if is_classification else REGRESSION_LABELS
    return features_dict, labels

  return _input_fn


class TensorForestEstimatorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._feature_columns = {
        feature_column.numeric_column('f_%d' % i, dtype=dtypes.float32)
        for i in range(NUM_FEATURES)
    }

  def _assert_checkpoint(self, model_dir, global_step, finalized_trees):
    self._assert_checkpoint_and_return_model(model_dir, global_step,
                                             finalized_trees)

  def _assert_checkpoint_and_return_model(self, model_dir, global_step,
                                          finalized_trees):
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

    return ensemble_proto

  def testInferBinaryClassifier(self):
    train_input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = tensor_forest.TensorForestClassifier(
        feature_columns=self._feature_columns,
        n_trees=1)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])


if __name__ == '__main__':
  googletest.main()
