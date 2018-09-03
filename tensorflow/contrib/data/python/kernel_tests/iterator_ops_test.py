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
"""Tests for experimental iterator_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util


class CheckpointInputPipelineHookTest(test.TestCase):

  @staticmethod
  def _model_fn(features, labels, mode, config):
    del labels
    del mode
    del config
    global_step = training_util.get_or_create_global_step()
    update_global_step_op = global_step.assign_add(1)
    latest_feature = variables.Variable(
        0, name='latest_feature', dtype=dtypes.int64)
    store_latest_feature_op = latest_feature.assign(features)
    ops.add_to_collection('my_vars', global_step)
    ops.add_to_collection('my_vars', latest_feature)
    return model_fn.EstimatorSpec(
        mode='train',
        train_op=control_flow_ops.group(
            [update_global_step_op, store_latest_feature_op]),
        loss=constant_op.constant(2.0))

  def _read_vars(self, model_dir):
    """Returns (global_step, latest_feature)."""
    with ops.Graph().as_default() as g:
      ckpt_path = checkpoint_management.latest_checkpoint(model_dir)
      meta_filename = ckpt_path + '.meta'
      saver_lib.import_meta_graph(meta_filename)
      saver = saver_lib.Saver()
      with self.session(graph=g) as sess:
        saver.restore(sess, ckpt_path)
        return sess.run(ops.get_collection('my_vars'))

  def _build_iterator_saver_hook(self, est):
    return iterator_ops.CheckpointInputPipelineHook(est)

  def testReturnDatasetFromInputFn(self):

    def _input_fn():
      return dataset_ops.Dataset.range(10)

    est = estimator.Estimator(model_fn=self._model_fn)

    est.train(_input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])
    self.assertSequenceEqual(self._read_vars(est.model_dir), (2, 1))
    est.train(_input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])
    self.assertSequenceEqual(self._read_vars(est.model_dir), (4, 3))

  def testBuildIteratorInInputFn(self):

    def _input_fn():
      ds = dataset_ops.Dataset.range(10)
      iterator = ds.make_one_shot_iterator()
      return iterator.get_next()

    est = estimator.Estimator(model_fn=self._model_fn)

    est.train(_input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])
    self.assertSequenceEqual(self._read_vars(est.model_dir), (2, 1))
    est.train(_input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])
    self.assertSequenceEqual(self._read_vars(est.model_dir), (4, 3))

  def testDoNotRestore(self):

    def _input_fn():
      return dataset_ops.Dataset.range(10)

    est = estimator.Estimator(model_fn=self._model_fn)

    est.train(_input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])
    self.assertSequenceEqual(self._read_vars(est.model_dir), (2, 1))
    est.train(_input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])
    self.assertSequenceEqual(self._read_vars(est.model_dir), (4, 3))
    # Hook not provided, input pipeline was not restored.
    est.train(_input_fn, steps=2)
    self.assertSequenceEqual(self._read_vars(est.model_dir), (6, 1))

  def testRaiseErrorIfNoIterator(self):

    def _input_fn():
      return constant_op.constant(1, dtype=dtypes.int64)

    est = estimator.Estimator(model_fn=self._model_fn)

    with self.assertRaises(ValueError):
      est.train(
          _input_fn, steps=2, hooks=[self._build_iterator_saver_hook(est)])


if __name__ == '__main__':
  test.main()
