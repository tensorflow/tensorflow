# pylint: disable=g-bad-file-header
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

"""Graph actions tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import tensorflow as tf

from tensorflow.contrib import testing
from tensorflow.contrib.learn.python import learn


class _Feeder(object):
  """Simple generator for `feed_fn`, returning 10 * step."""

  def __init__(self, tensor):
    self._step = 0
    self._tensor = tensor

  @property
  def step(self):
    return self._step

  def feed_fn(self):
    value = self._step * 10.0
    self._step += 1
    return {self._tensor: value}


class GraphActionsTest(tf.test.TestCase):
  """Graph actions tests."""

  def setUp(self):
    learn.graph_actions.clear_summary_writers()
    self._output_dir = tempfile.mkdtemp()
    testing.FakeSummaryWriter.install()

  def tearDown(self):
    testing.FakeSummaryWriter.uninstall()
    if self._output_dir:
      shutil.rmtree(self._output_dir)
    learn.graph_actions.clear_summary_writers()

  def _assert_summaries(
      self, output_dir, expected_summaries=None, expected_graphs=None,
      expected_session_logs=None):
    writer = learn.graph_actions.get_summary_writer(output_dir)
    self.assertTrue(isinstance(writer, testing.FakeSummaryWriter))
    writer.assert_summaries(
        self, expected_logdir=output_dir, expected_graph=tf.get_default_graph(),
        expected_summaries=expected_summaries,
        expected_added_graphs=expected_graphs,
        expected_session_logs=expected_session_logs)

  # TODO(ptucker): Test lock, multi-threaded access?
  def test_summary_writer(self):
    self._assert_summaries('log/dir/0')
    self.assertTrue(
        learn.graph_actions.get_summary_writer('log/dir/0') is
        learn.graph_actions.get_summary_writer('log/dir/0'))
    self.assertTrue(
        learn.graph_actions.get_summary_writer('log/dir/0') is not
        learn.graph_actions.get_summary_writer('log/dir/1'))

  # TODO(ptucker): Test restore_checkpoint_path for eval.
  # TODO(ptucker): Test start_queue_runners for both eval & train.
  # TODO(ptucker): Test coord.request_stop & coord.join for eval.

  def _build_inference_graph(self):
    """Build simple inference graph.

    This includes a regular variable, local variable, and fake table.

    Returns:
      Tuple of 3 `Tensor` objects, 2 input and 1 output.
    """
    tf.contrib.framework.create_global_step()
    in0 = tf.Variable(1.0)
    in1 = tf.contrib.framework.local_variable(2.0)
    fake_table = tf.Variable(
        3.0, trainable=False, collections=['fake_tables'],
        name='fake_table_var')
    in0.graph.add_to_collections(
        [tf.GraphKeys.TABLE_INITIALIZERS], fake_table.initializer)
    out = in0 + in1 + fake_table
    return in0, in1, out

  def test_infer(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      in0, in1, out = self._build_inference_graph()
      self.assertEqual(
          {'a': 1.0, 'b': 2.0, 'c': 6.0},
          learn.graph_actions.infer(None, {'a': in0, 'b': in1, 'c': out}))

  def test_infer_different_default_graph(self):
    with self.test_session():
      with tf.Graph().as_default():
        in0, in1, out = self._build_inference_graph()
      with tf.Graph().as_default():
        self.assertEqual(
            {'a': 1.0, 'b': 2.0, 'c': 6.0},
            learn.graph_actions.infer(None, {'a': in0, 'b': in1, 'c': out}))

  def test_infer_invalid_feed(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      in0, _, _ = self._build_inference_graph()
      with self.assertRaisesRegexp(
          tf.errors.InvalidArgumentError, 'both fed and fetched'):
        learn.graph_actions.infer(None, {'a': in0}, feed_dict={in0: 4.0})

  def test_infer_feed(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      in0, _, out = self._build_inference_graph()
      self.assertEqual(
          {'c': 9.0},
          learn.graph_actions.infer(None, {'c': out}, feed_dict={in0: 4.0}))

  # TODO(ptucker): Test saver and ckpt_path for eval & train.
  # TODO(ptucker): Test eval for 1 epoch.

  def test_evaluate_invalid_args(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.evaluate(
            g, output_dir=None, checkpoint_path=None,
            eval_dict={'a': tf.constant(1.0)})
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.evaluate(
            g, output_dir='', checkpoint_path=None,
            eval_dict={'a': tf.constant(1.0)})

  def test_evaluate(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      _, _, out = self._build_inference_graph()
      self._assert_summaries(self._output_dir)
      results = learn.graph_actions.evaluate(
          g, output_dir=self._output_dir, checkpoint_path=None,
          eval_dict={'a': out}, max_steps=1)
      self.assertEqual(({'a': 6.0}, 0), results)
      self._assert_summaries(
          self._output_dir, expected_summaries={0: {'a': 6.0}})

  def test_evaluate_feed_fn(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      in0, _, out = self._build_inference_graph()
      self._assert_summaries(self._output_dir)
      feeder = _Feeder(in0)
      results = learn.graph_actions.evaluate(
          g, output_dir=self._output_dir, checkpoint_path=None,
          eval_dict={'a': out}, feed_fn=feeder.feed_fn, max_steps=3)
      self.assertEqual(3, feeder.step)
      self.assertEqual(({'a': 25.0}, 0), results)
      self._assert_summaries(
          self._output_dir, expected_summaries={0: {'a': 25.0}})

  def test_train_invalid_args(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      train_op = tf.constant(1.0)
      loss_op = tf.constant(2.0)
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.train(
            g, output_dir=None, train_op=train_op, loss_op=loss_op)
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.train(
            g, output_dir='', train_op=tf.constant(1.0),
            loss_op=tf.constant(2.0))
      with self.assertRaisesRegexp(ValueError, 'train_op'):
        learn.graph_actions.train(
            g, output_dir=self._output_dir, train_op=None, loss_op=loss_op)
      with self.assertRaisesRegexp(ValueError, 'loss_op'):
        learn.graph_actions.train(
            g, output_dir=self._output_dir, train_op=tf.constant(1.0),
            loss_op=None)
      with self.assertRaisesRegexp(ValueError, 'global_step'):
        learn.graph_actions.train(
            g, output_dir=self._output_dir, train_op=tf.constant(1.0),
            loss_op=loss_op)

  # TODO(ptucker): Resume training from previous ckpt.
  # TODO(ptucker): !supervisor_is_chief
  # TODO(ptucker): Custom init op for training.

  def _expected_train_session_logs(self):
    return [
        tf.SessionLog(status=tf.SessionLog.START),
        tf.SessionLog(
            status=tf.SessionLog.CHECKPOINT,
            checkpoint_path='%s/model.ckpt' % self._output_dir),
    ]

  def test_train(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      with tf.control_dependencies(self._build_inference_graph()):
        train_op = tf.assign_add(tf.contrib.framework.get_global_step(), 1)
      self._assert_summaries(self._output_dir)
      loss = learn.graph_actions.train(
          g, output_dir=self._output_dir, train_op=train_op,
          loss_op=tf.constant(2.0), steps=1)
      self.assertEqual(2.0, loss)
      self._assert_summaries(
          self._output_dir,
          expected_graphs=[g],
          expected_session_logs=self._expected_train_session_logs())

  def test_train_loss(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      tf.contrib.framework.create_global_step()
      loss_var = tf.contrib.framework.local_variable(10.0)
      train_op = tf.group(
          tf.assign_add(tf.contrib.framework.get_global_step(), 1),
          tf.assign_add(loss_var, -1.0))
      self._assert_summaries(self._output_dir)
      loss = learn.graph_actions.train(
          g, output_dir=self._output_dir, train_op=train_op,
          loss_op=loss_var.value(), steps=6)
      self.assertEqual(4.0, loss)
      self._assert_summaries(
          self._output_dir,
          expected_graphs=[g],
          expected_session_logs=self._expected_train_session_logs())

  def test_train_summaries(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      with tf.control_dependencies(self._build_inference_graph()):
        train_op = tf.assign_add(tf.contrib.framework.get_global_step(), 1)
      loss_op = tf.constant(2.0)
      tf.scalar_summary('loss', loss_op)
      self._assert_summaries(self._output_dir)
      loss = learn.graph_actions.train(
          g, output_dir=self._output_dir, train_op=train_op, loss_op=loss_op,
          steps=1)
      self.assertEqual(2.0, loss)
      self._assert_summaries(
          self._output_dir,
          expected_graphs=[g],
          expected_session_logs=self._expected_train_session_logs(),
          expected_summaries={1: {'loss': 2.0}})


if __name__ == '__main__':
  tf.test.main()
