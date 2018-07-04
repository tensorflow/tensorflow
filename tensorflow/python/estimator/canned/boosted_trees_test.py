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
"""Tests boosted_trees estimators and model_fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.canned import boosted_trees
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import session_run_hook

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

# EXAMPLE_ID is not exposed to Estimator yet, but supported at model_fn level.
EXAMPLE_IDS = np.array([0, 1, 2, 3, 4], dtype=np.int64)
EXAMPLE_ID_COLUMN = '__example_id__'


def _make_train_input_fn(is_classification):
  """Makes train input_fn for classification/regression."""

  def _input_fn():
    features_dict = dict(FEATURES_DICT)  # copies the dict to add an entry.
    features_dict[EXAMPLE_ID_COLUMN] = constant_op.constant(EXAMPLE_IDS)
    labels = CLASSIFICATION_LABELS if is_classification else REGRESSION_LABELS
    return features_dict, labels

  return _input_fn


def _make_train_input_fn_dataset(is_classification, batch=None, repeat=None):
  """Makes input_fn using Dataset."""

  def _input_fn():
    features_dict = dict(FEATURES_DICT)  # copies the dict to add an entry.
    features_dict[EXAMPLE_ID_COLUMN] = constant_op.constant(EXAMPLE_IDS)
    labels = CLASSIFICATION_LABELS if is_classification else REGRESSION_LABELS
    if batch:
      ds = dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensor_slices(features_dict),
           dataset_ops.Dataset.from_tensor_slices(labels))).batch(batch)
    else:
      ds = dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensors(features_dict),
           dataset_ops.Dataset.from_tensors(labels)))
    # repeat indefinitely by default, or stop at the given step.
    ds = ds.repeat(repeat)
    return ds

  return _input_fn


class BoostedTreesEstimatorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._feature_columns = {
        feature_column.bucketized_column(
            feature_column.numeric_column('f_%d' % i, dtype=dtypes.float32),
            BUCKET_BOUNDARIES)
        for i in range(NUM_FEATURES)
    }

  def _assert_checkpoint(self, model_dir, global_step, finalized_trees,
                         attempted_layers):
    self._assert_checkpoint_and_return_model(model_dir, global_step,
                                             finalized_trees, attempted_layers)

  def _assert_checkpoint_and_return_model(self, model_dir, global_step,
                                          finalized_trees, attempted_layers):
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

  def testFirstCheckpointWorksFine(self):
    """Tests that eval/pred doesn't crash with the very first checkpoint.

    The step-0 checkpoint will have only an empty ensemble, and a separate eval
    job might read from it and crash.
    This test ensures that prediction/evaluation works fine with it.
    """
    input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    class BailOutWithoutTraining(session_run_hook.SessionRunHook):

      def before_run(self, run_context):
        raise StopIteration('to bail out.')

    est.train(input_fn, steps=100,  # must stop at 0 anyway.
              hooks=[BailOutWithoutTraining()])
    self._assert_checkpoint(
        est.model_dir, global_step=0, finalized_trees=0, attempted_layers=0)
    # Empty ensemble returns 0 logits, so that all output labels are 0.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 0.6)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [0], [0], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testTrainAndEvaluateBinaryClassifier(self):
    input_fn = _make_train_input_fn(is_classification=True)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  def testInferBinaryClassifier(self):
    train_input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testTrainClassifierWithRankOneLabel(self):
    """Tests that label with rank-1 tensor is also accepted by classifier."""
    def _input_fn_with_rank_one_label():
      return FEATURES_DICT, [0., 1., 1., 0., 0.]

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(_input_fn_with_rank_one_label, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_rank_one_label, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  def testTrainClassifierWithLabelVocabulary(self):
    apple, banana = 'apple', 'banana'
    def _input_fn_with_label_vocab():
      return FEATURES_DICT, [[apple], [banana], [banana], [apple], [apple]]
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        label_vocabulary=[apple, banana])
    est.train(input_fn=_input_fn_with_label_vocab, steps=5)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_label_vocab, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testTrainClassifierWithIntegerLabel(self):
    def _input_fn_with_integer_label():
      return (FEATURES_DICT,
              constant_op.constant([[0], [1], [1], [0], [0]], dtypes.int32))
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(input_fn=_input_fn_with_integer_label, steps=5)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_integer_label, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testTrainClassifierWithDataset(self):
    train_input_fn = _make_train_input_fn_dataset(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  def testTrainAndEvaluateRegressor(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5)

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 1.008551)

  def testInferRegressor(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  def testTrainRegressorWithRankOneLabel(self):
    """Tests that label with rank-1 tensor is also accepted by regressor."""
    def _input_fn_with_rank_one_label():
      return FEATURES_DICT, [1.5, 0.3, 0.2, 2., 5.]

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(_input_fn_with_rank_one_label, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_rank_one_label, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)

  def testTrainRegressorWithDataset(self):
    train_input_fn = _make_train_input_fn_dataset(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  def testTrainRegressorWithDatasetBatch(self):
    # The batch_size as the entire data size should yield the same result as
    # dataset without batching.
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, batch=5)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  def testTrainRegressorWithDatasetLargerBatch(self):
    # The batch_size as the multiple of the entire data size should still yield
    # the same result.
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, batch=15)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  def testTrainRegressorWithDatasetSmallerBatch(self):
    # Even when using small batches, if (n_batches_per_layer * batch_size) makes
    # the same entire data size, the result should be the same.
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, batch=1)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=5,
        n_trees=1,
        max_depth=5)
    # Train stops after (n_batches_per_layer * n_trees * max_depth) steps.
    est.train(train_input_fn, steps=100)
    self._assert_checkpoint(
        est.model_dir, global_step=25, finalized_trees=1, attempted_layers=5)
    # 5 batches = one epoch.
    eval_res = est.evaluate(input_fn=train_input_fn, steps=5)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  def testTrainRegressorWithDatasetWhenInputIsOverEarlier(self):
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, repeat=3)  # to stop input after 3 steps.
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    # Note that training will stop when input exhausts.
    # This might not be a typical pattern, but dataset.repeat(3) causes
    # the input stream to cease after 3 steps.
    est.train(train_input_fn, steps=100)
    self._assert_checkpoint(
        est.model_dir, global_step=3, finalized_trees=0, attempted_layers=3)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 3.777295)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.353850], [0.254100], [0.106850], [0.712100], [1.012100]],
        [pred['predictions'] for pred in predictions])

  def testTrainEvaluateAndPredictWithIndicatorColumn(self):
    categorical = feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))
    feature_indicator = feature_column.indicator_column(categorical)
    bucketized_col = feature_column.bucketized_column(
        feature_column.numeric_column(
            'an_uninformative_feature', dtype=dtypes.float32),
        BUCKET_BOUNDARIES)

    labels = np.array([[0.], [5.7], [5.7], [0.], [0.]], dtype=np.float32)
    # Our categorical feature defines the labels perfectly
    input_fn = numpy_io.numpy_input_fn(
        x={
            'an_uninformative_feature': np.array([1, 1, 1, 1, 1]),
            'categorical': np.array(['bad', 'good', 'good', 'ok', 'bad']),
        },
        y=labels,
        batch_size=5,
        shuffle=False)

    # Train depth 1 tree.
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[bucketized_col, feature_indicator],
        n_batches_per_layer=1,
        n_trees=1,
        learning_rate=1.0,
        max_depth=1)

    num_steps = 1
    est.train(input_fn, steps=num_steps)
    ensemble = self._assert_checkpoint_and_return_model(
        est.model_dir, global_step=1, finalized_trees=1, attempted_layers=1)

    # We learnt perfectly.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['loss'], 0)

    predictions = list(est.predict(input_fn))
    self.assertAllClose(
        labels,
        [pred['predictions'] for pred in predictions])

    self.assertEqual(3, len(ensemble.trees[0].nodes))

    # Check that the split happened on 'good' value, which will be encoded as
    # feature with index 2 (0-numeric, 1 - 'bad')
    self.assertEqual(2, ensemble.trees[0].nodes[0].bucketized_split.feature_id)
    self.assertEqual(0, ensemble.trees[0].nodes[0].bucketized_split.threshold)

  def testTrainEvaluateAndPredictWithOnlyIndicatorColumn(self):
    categorical = feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))
    feature_indicator = feature_column.indicator_column(categorical)

    labels = np.array([[0.], [5.7], [5.7], [0.], [0.]], dtype=np.float32)
    # Our categorical feature defines the labels perfectly
    input_fn = numpy_io.numpy_input_fn(
        x={
            'categorical': np.array(['bad', 'good', 'good', 'ok', 'bad']),
        },
        y=labels,
        batch_size=5,
        shuffle=False)

    # Train depth 1 tree.
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[feature_indicator],
        n_batches_per_layer=1,
        n_trees=1,
        learning_rate=1.0,
        max_depth=1)

    num_steps = 1
    est.train(input_fn, steps=num_steps)
    ensemble = self._assert_checkpoint_and_return_model(
        est.model_dir, global_step=1, finalized_trees=1, attempted_layers=1)

    # We learnt perfectly.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['loss'], 0)

    predictions = list(est.predict(input_fn))
    self.assertAllClose(
        labels,
        [pred['predictions'] for pred in predictions])

    self.assertEqual(3, len(ensemble.trees[0].nodes))

    # Check that the split happened on 'good' value, which will be encoded as
    # feature with index 1 (0 - 'bad', 2 - 'ok')
    self.assertEqual(1, ensemble.trees[0].nodes[0].bucketized_split.feature_id)
    self.assertEqual(0, ensemble.trees[0].nodes[0].bucketized_split.threshold)


class ModelFnTests(test_util.TensorFlowTestCase):
  """Tests bt_model_fn including unexposed internal functionalities."""

  def setUp(self):
    self._feature_columns = {
        feature_column.bucketized_column(
            feature_column.numeric_column('f_%d' % i, dtype=dtypes.float32),
            BUCKET_BOUNDARIES) for i in range(NUM_FEATURES)
    }

  def _get_expected_ensembles_for_classification(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.0625
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.105518
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.224091
            }
          }
          nodes {
            leaf {
              scalar: 0.056815
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.105518
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.224091
            }
          }
          nodes {
            leaf {
              scalar: 0.056815
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.287131
            }
          }
          nodes {
            leaf {
              scalar: 0.162042
            }
          }
          nodes {
            leaf {
              scalar: -0.086986
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round)

  def _get_expected_ensembles_for_classification_with_bias(self):
    first_round = """
        trees {
          nodes {
            leaf {
              scalar: -0.405086
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.407711
              original_leaf {
                scalar: -0.405086
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.556054
            }
          }
          nodes {
            leaf {
              scalar: -0.301233
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.407711
              original_leaf {
                scalar: -0.405086
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              original_leaf {
                scalar: -0.556054
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.09876
              original_leaf {
                scalar: -0.301233
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.698072
            }
          }
          nodes {
            leaf {
              scalar: -0.556054
            }
          }
          nodes {
            leaf {
              scalar: -0.106016
            }
          }
          nodes {
            leaf {
              scalar: -0.27349
            }
          }
        }
        trees {
          nodes {
            leaf {
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_end: 1
        }
        """
    forth_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.4077113
              original_leaf {
                scalar: -0.405086
              }
            }
          }
          nodes {
            bucketized_split {
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              original_leaf {
                scalar: -0.556054
              }
            }
          }
          nodes {
            bucketized_split {
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.09876
              original_leaf {
                scalar: -0.301233
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.698072
            }
          }
          nodes {
            leaf {
              scalar: -0.556054
            }
          }
          nodes {
            leaf {
              scalar: -0.106016
            }
          }
          nodes {
            leaf {
              scalar: -0.27349
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.289927
            }
          }
          nodes {
            leaf {
              scalar: -0.134588
            }
          }
          nodes {
            leaf {
              scalar: 0.083838            
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round, forth_round)

  def _get_expected_ensembles_for_regression(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169714
            }
          }
          nodes {
            leaf {
              scalar: 0.241322
            }
          }
          nodes {
            leaf {
              scalar: 0.083951
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169714
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.673407
              original_leaf {
                scalar: 0.241322
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.324102
              original_leaf {
                scalar: 0.083951
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.563167
            }
          }
          nodes {
            leaf {
              scalar: 0.247047
            }
          }
          nodes {
            leaf {
              scalar: 0.095273
            }
          }
          nodes {
            leaf {
              scalar: 0.222102
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169714
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.673407
              original_leaf {
                scalar: 0.241322
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.324102
              original_leaf {
                scalar: 0.083951
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.563167
            }
          }
          nodes {
            leaf {
              scalar: 0.247047
            }
          }
          nodes {
            leaf {
              scalar: 0.095273
            }
          }
          nodes {
            leaf {
              scalar: 0.222102
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.981026
            }
          }
          nodes {
            leaf {
              scalar: 0.005166
            }
          }
          nodes {
            leaf {
              scalar: 0.180281
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round)

  def _get_expected_ensembles_for_regression_with_bias(self):
    first_round = """
        trees {
          nodes {
            leaf {
              scalar: 1.799974
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.190442
              original_leaf {
                scalar: 1.799974
              }
            }
          }
          nodes {
            leaf {
              scalar: 1.862786
            }
          }
          nodes {
            leaf {
              scalar: 1.706149
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.190442
              original_leaf {
                scalar: 1.799974
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.683594
              original_leaf {
                scalar: 1.862786
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.322693
              original_leaf {
                scalar: 1.706149
              }
            }
          }
          nodes {
            leaf {
              scalar: 2.024487
            }
          }
          nodes {
            leaf {
              scalar: 1.710319
            }
          }
          nodes {
            leaf {
              scalar: 1.559208
            }
          }
          nodes {
            leaf {
              scalar: 1.686037
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    forth_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.190442
              original_leaf {
                scalar:  1.799974
              }
            }
          }
          nodes {
            bucketized_split {
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.683594
              original_leaf {
                scalar: 1.8627863
              }
            }
          }
          nodes {
            bucketized_split {
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.322693
              original_leaf {
                scalar: 1.706149
              }
            }
          }
          nodes {
            leaf {
              scalar: 2.024487
            }
          }
          nodes {
            leaf {
              scalar: 1.710319
            }
          }
          nodes {
            leaf {
              scalar: 1.5592078
            }
          }
          nodes {
            leaf {
              scalar: 1.686037
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.972589
            }
          }
          nodes {
            leaf {
              scalar: -0.137592
            }
          }
          nodes {
            leaf {
              scalar: 0.034926
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round, forth_round)

  def _get_train_op_and_ensemble(self,
                                 head,
                                 config,
                                 is_classification,
                                 train_in_memory,
                                 center_bias=False):
    """Calls bt_model_fn() and returns the train_op and ensemble_serialzed."""
    features, labels = _make_train_input_fn(is_classification)()

    tree_hparams = boosted_trees._TreeHParams(  # pylint:disable=protected-access
        n_trees=2,
        max_depth=2,
        learning_rate=0.1,
        l1=0.,
        l2=0.01,
        tree_complexity=0.,
        min_node_weight=0.,
        center_bias=center_bias)

    estimator_spec = boosted_trees._bt_model_fn(  # pylint:disable=protected-access
        features=features,
        labels=labels,
        mode=model_fn.ModeKeys.TRAIN,
        head=head,
        feature_columns=self._feature_columns,
        tree_hparams=tree_hparams,
        example_id_column_name=EXAMPLE_ID_COLUMN,
        n_batches_per_layer=1,
        config=config,
        train_in_memory=train_in_memory)
    resources.initialize_resources(resources.shared_resources()).run()
    variables.global_variables_initializer().run()
    variables.local_variables_initializer().run()

    # Gets the train_op and serialized proto of the ensemble.
    shared_resources = resources.shared_resources()
    self.assertEqual(1, len(shared_resources))
    train_op = estimator_spec.train_op
    with ops.control_dependencies([train_op]):
      _, ensemble_serialized = (
          gen_boosted_trees_ops.boosted_trees_serialize_ensemble(
              shared_resources[0].handle))
    return train_op, ensemble_serialized

  def testTrainClassifierInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_classification())
    with self.test_session() as sess:
      # Train with train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_classification_head(n_classes=2),
            run_config.RunConfig(),
            is_classification=True,
            train_in_memory=True)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainClassifierWithCenterBiasInMemory(self):
    ops.reset_default_graph()

    # When bias centering is on, we expect the very first node to have the
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_classification_with_bias())

    with self.test_session() as sess:
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_classification_head(n_classes=2),
            run_config.RunConfig(),
            is_classification=True,
            train_in_memory=True,
            center_bias=True)

      # 4 iterations to center bias.
      for _ in range(4):
        _, serialized = sess.run([train_op, ensemble_serialized])

      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)

      self.assertProtoEquals(expected_forth, ensemble_proto)

  def testTrainClassifierNonInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_classification())
    with self.test_session() as sess:
      # Train without train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_classification_head(n_classes=2),
            run_config.RunConfig(),
            is_classification=True,
            train_in_memory=False)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainClassifierWithCenterBiasNonInMemory(self):
    ops.reset_default_graph()

    # When bias centering is on, we expect the very first node to have the
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_classification_with_bias())

    with self.test_session() as sess:
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_classification_head(n_classes=2),
            run_config.RunConfig(),
            is_classification=True,
            train_in_memory=False,
            center_bias=True)
      # 4 iterations to center bias.
      for _ in range(4):
        _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_forth, ensemble_proto)

  def testTrainRegressorInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_regression())
    with self.test_session() as sess:
      # Train with train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_regression_head(label_dimension=1),
            run_config.RunConfig(),
            is_classification=False,
            train_in_memory=True)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainRegressorInMemoryWithCenterBias(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_regression_with_bias())
    with self.test_session() as sess:
      # Train with train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_regression_head(label_dimension=1),
            run_config.RunConfig(),
            is_classification=False,
            train_in_memory=True,
            center_bias=True)
      # 3 iterations to center bias.
      for _ in range(3):
        _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)

      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_forth, ensemble_proto)

  def testTrainRegressorNonInMemory(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_regression())
    with self.test_session() as sess:
      # Train without train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_regression_head(label_dimension=1),
            run_config.RunConfig(),
            is_classification=False,
            train_in_memory=False)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainRegressorNotInMemoryWithCenterBias(self):
    ops.reset_default_graph()
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_regression_with_bias())
    with self.test_session() as sess:
      # Train with train_in_memory mode.
      with sess.graph.as_default():
        train_op, ensemble_serialized = self._get_train_op_and_ensemble(
            boosted_trees._create_regression_head(label_dimension=1),
            run_config.RunConfig(),
            is_classification=False,
            train_in_memory=False,
            center_bias=True)
      # 3 iterations to center the bias (because we are using regularization).
      for _ in range(3):
        _, serialized = sess.run([train_op, ensemble_serialized])

      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_forth, ensemble_proto)


if __name__ == '__main__':
  googletest.main()
