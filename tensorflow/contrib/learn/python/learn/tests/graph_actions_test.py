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

import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.python.training import summary_io


# TODO(ptucker): Replace with mock framework.
class _FakeSummaryWriter(object):

  def __init__(self, logdir, graph=None):
    self._logdir = logdir
    self._graph = graph
    self._summaries = {}
    self._flushed = True

  @property
  def logdir(self):
    return self._logdir

  @property
  def graph(self):
    return self._graph

  @property
  def summaries(self):
    return self._summaries

  @property
  def flushed(self):
    return self._flushed

  def add_summary(self, summary, current_global_step):
    if current_global_step in self._summaries:
      raise ValueError('Dupe summary for step %s.' % current_global_step)
    self._summaries[current_global_step] = summary
    self._flushed = False

  def flush(self):
    self._flushed = True


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
    self._original_summary_writer = summary_io.SummaryWriter
    summary_io.SummaryWriter = _FakeSummaryWriter

  def tearDown(self):
    summary_io.SummaryWriter = self._original_summary_writer
    learn.graph_actions.clear_summary_writers()

  def _assert_fake_summary_writer(self, output_dir, expected_summaries=None):
    writer = learn.graph_actions.get_summary_writer(output_dir)
    self.assertTrue(isinstance(writer, _FakeSummaryWriter))
    self.assertEqual(output_dir, writer.logdir)
    self.assertTrue(tf.get_default_graph() is writer.graph)
    self.assertTrue(writer.flushed)
    expected_summaries = expected_summaries or {}
    expected_steps = expected_summaries.keys()
    self.assertEqual(set(expected_steps), set(writer.summaries.keys()))
    for step in expected_steps:
      actual_simple_values = {}
      for v in writer.summaries[step].value:
        actual_simple_values[v.tag] = v.simple_value
      self.assertEqual(expected_summaries[step], actual_simple_values)

  # TODO(ptucker): Test lock, multi-threaded access?
  def test_summary_writer(self):
    self._assert_fake_summary_writer('log/dir/0')
    self.assertTrue(
        learn.graph_actions.get_summary_writer('log/dir/0') is
        learn.graph_actions.get_summary_writer('log/dir/0'))
    self.assertTrue(
        learn.graph_actions.get_summary_writer('log/dir/0') is not
        learn.graph_actions.get_summary_writer('log/dir/1'))

  # TODO(ptucker): Test restore_checkpoint_path.
  # TODO(ptucker): Test start_queue_runners.
  # TODO(ptucker): Test coord.request_stop & coord.join.

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

  # TODO(ptucker): Test saver and ckpt_path.
  # TODO(ptucker): Test eval for 1 epoch.

  def test_evaluate(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      _, _, out = self._build_inference_graph()
      output_dir = 'out/dir'
      self._assert_fake_summary_writer(output_dir, {})
      results = learn.graph_actions.evaluate(
          g, output_dir=output_dir, checkpoint_path=None, eval_dict={'a': out},
          max_steps=1)
      self.assertEqual(({'a': 6.0}, 0), results)
      self._assert_fake_summary_writer(output_dir, {0: {'a': 6.0}})

  def test_evaluate_feed_fn(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      in0, _, out = self._build_inference_graph()
      output_dir = 'out/dir'
      self._assert_fake_summary_writer(output_dir, {})
      feeder = _Feeder(in0)
      results = learn.graph_actions.evaluate(
          g, output_dir=output_dir, checkpoint_path=None, eval_dict={'a': out},
          feed_fn=feeder.feed_fn, max_steps=3)
      self.assertEqual(3, feeder.step)
      self.assertEqual(({'a': 25.0}, 0), results)
      self._assert_fake_summary_writer(output_dir, {0: {'a': 25.0}})


if __name__ == '__main__':
  tf.test.main()
