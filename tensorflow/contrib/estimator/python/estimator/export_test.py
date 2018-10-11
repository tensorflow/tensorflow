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
"""Tests for contrib wrapping of export_saved_model_for_mode functionality.

These are direct copies of the tests included in core, with import locations
changed. These should be removed when the functionality in core is part of the
public API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.contrib.estimator.python.estimator import export as contrib_export
from tensorflow.python.client import session
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import training
from tensorflow.python.util import compat


def _model_fn_for_export_tests(features, labels, mode):
  _, _ = features, labels
  variables.Variable(1., name='weight')
  scores = constant_op.constant([3.])
  classes = constant_op.constant(['wumpus'])
  update_global_step = state_ops.assign_add(training.get_global_step(), 1)
  with ops.control_dependencies([update_global_step]):
    train_op = constant_op.constant(2.)
  return model_fn_lib.EstimatorSpec(
      mode,
      predictions=constant_op.constant(10.),
      loss=constant_op.constant(1.),
      train_op=train_op,
      export_outputs={
          'test': export_output.ClassificationOutput(scores, classes)})


def _x_y_input_fn():
  return ({'x': constant_op.constant([[1], [1]]),
           'y': constant_op.constant([[2], [2]])},
          constant_op.constant([[1], [1]]))


def _model_fn_with_x_y(features, labels, mode):
  _ = labels
  variables.Variable(1., name='weight')
  scores = constant_op.constant([3.])
  classes = constant_op.constant(['wumpus'])
  if mode == model_fn_lib.ModeKeys.PREDICT:
    variables.Variable(36., name='name_collision')
    return model_fn_lib.EstimatorSpec(
        mode,
        predictions=constant_op.constant(10.),
        export_outputs={
            'test': export_output.ClassificationOutput(scores, classes)})
  else:
    prefix = 'eval_' if mode == model_fn_lib.ModeKeys.EVAL else ''

    multiplied = math_ops.multiply(
        features['x'], features['y'], name='{}multiplied'.format(prefix))
    metrics = {'mean': metrics_lib.mean(features['x'] - features['y'],
                                        name='{}mean'.format(prefix))}
    variables.Variable(1., name='later_var')
    variables.Variable(3., name='name_collision')
    return model_fn_lib.EstimatorSpec(
        mode,
        predictions=multiplied,
        loss=constant_op.constant(1.),
        train_op=state_ops.assign_add(training.get_global_step(), 1),
        eval_metric_ops=metrics)


def _get_serving_input_receiver_fn():
  feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                  'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
  return export.build_parsing_serving_input_receiver_fn(feature_spec)


def _get_supervised_input_receiver_fn():
  feature_spec = {
      'x': array_ops.placeholder(
          dtype=dtypes.int64, shape=(2, 1), name='feature_x'),
      'y': array_ops.placeholder(
          dtype=dtypes.int64, shape=(2, 1), name='feature_y')
      }
  label_spec = array_ops.placeholder(
      dtype=dtypes.float32, shape=[1], name='truth')

  return export.build_raw_supervised_input_receiver_fn(
      feature_spec, label_spec)


class EstimatorExportTest(test.TestCase):

  def test_export_saved_model_train(self):
    self._test_export_saved_model_for_mode(
        _get_supervised_input_receiver_fn(), model_fn_lib.ModeKeys.TRAIN)

  def test_export_saved_model_eval(self):
    self._test_export_saved_model_for_mode(
        _get_supervised_input_receiver_fn(), model_fn_lib.ModeKeys.EVAL)

  def test_export_saved_model_predict(self):
    self._test_export_saved_model_for_mode(
        _get_serving_input_receiver_fn(), model_fn_lib.ModeKeys.PREDICT)

  def _test_export_saved_model_for_mode(self, input_receiver_fn, mode):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = contrib_export.export_saved_model_for_mode(
        est, export_dir_base, input_receiver_fn, mode=mode)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    tag_set = model_fn_lib.EXPORT_TAG_MAP[mode]
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, tag_set, export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertFalse('name_collision_1' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_receiver_map(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExample' in graph_ops)
        self.assertFalse('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_train_only(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('multiplied' in graph_ops)
        self.assertTrue('mean/update_op' in graph_ops)
        self.assertFalse('eval_multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_eval_only(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.EVAL], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('eval_multiplied' in graph_ops)
        self.assertTrue('eval_mean/value' in graph_ops)
        self.assertFalse('multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_no_serving(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('multiplied' in graph_ops)
        self.assertFalse('eval_multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.EVAL], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('eval_multiplied' in graph_ops)
        self.assertFalse('multiplied' in graph_ops)
        # TODO(karmel): is this the desired behavior when names are shared?
        self.assertTrue('feature_x_1' in graph_ops)
        self.assertTrue('feature_y_1' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_three_defs(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    # Restore, to validate that the export was well-formed.
    for tag_set in model_fn_lib.EXPORT_TAG_MAP.values():
      with ops.Graph().as_default() as graph:
        with session.Session(graph=graph) as sess:
          loader.load(sess, tag_set, export_dir)
          graph_ops = [x.name for x in graph.get_operations()]
          self.assertTrue('global_step/Assign' in graph_ops)
          self.assertTrue('global_step/Initializer/zeros' in graph_ops)
          self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_all_vars(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('later_var' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertFalse('later_var' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_name_collision(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('name_collision' in graph_ops)
        self.assertFalse('name_collision_1' in graph_ops)
        collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        self.assertEqual(3, collection_vars[-1].eval())

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('name_collision' in graph_ops)
        self.assertFalse('name_collision_1' in graph_ops)
        collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        # This is a non-obvious detail: when we load the estimator spec
        # for predict, name_collision gets set to 36. However, we then restore
        # from checkpoint, which should overwrite that var and make it the 3
        # from training. In practice, this would not be a good way to write
        # a model_fn, but leaving this check in for now to ensure consistency
        # with what would happen given our current order of spec, then
        # checkpoint.
        self.assertEqual(3, collection_vars[-1].eval())

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def _test_export_all_saved_models(self, input_receiver_fn_map):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_with_x_y)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = contrib_export.export_all_saved_models(
        est, export_dir_base, input_receiver_fn_map)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))

    self._validate_exported_files(export_dir)

    return export_dir, tmpdir

  def _validate_exported_files(self, export_dir):
    self.assertTrue(gfile.Exists(export_dir))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('saved_model.pb'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.index'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.data-00000-of-00001'))))


if __name__ == '__main__':
  test.main()
