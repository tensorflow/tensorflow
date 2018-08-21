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

from tensorflow.contrib import testing
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.monitors import BaseMonitor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib


class _Feeder(object):
  """Simple generator for `feed_fn`, returning 10 * step."""

  def __init__(self, tensor, max_step):
    self._step = 0
    self._tensor = tensor
    self._max_step = max_step

  @property
  def step(self):
    return self._step

  def feed_fn(self):
    if self._step >= self._max_step:
      raise StopIteration
    value = self._step * 10.0
    self._step += 1
    return {self._tensor: value}


class _BaseMonitorWrapper(BaseMonitor):
  """Base monitor wrapper to facilitate testing.

  This monitor can act as either chief-exclusive or non-exclusive.
  """

  def __init__(self, run_on_all_workers):
    super(_BaseMonitorWrapper, self).__init__()
    self._run_on_all_workers = run_on_all_workers
    self._is_active = False
    self._has_step = False

  @property
  def run_on_all_workers(self):
    return self._run_on_all_workers

  @property
  def is_active(self):
    return self._is_active

  @property
  def has_step(self):
    return self._has_step

  def begin(self, max_steps=None):
    self._is_active = True
    return super(_BaseMonitorWrapper, self).begin(max_steps)

  def step_begin(self, step):
    self._has_step = True
    return super(_BaseMonitorWrapper, self).step_begin(step)


class GraphActionsTest(test.TestCase):
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

  def _assert_summaries(self,
                        output_dir,
                        writer,
                        expected_summaries=None,
                        expected_graphs=None,
                        expected_meta_graphs=None,
                        expected_session_logs=None):
    self.assertTrue(isinstance(writer, testing.FakeSummaryWriter))
    writer.assert_summaries(
        self,
        expected_logdir=output_dir,
        expected_graph=ops.get_default_graph(),
        expected_summaries=expected_summaries,
        expected_added_graphs=expected_graphs,
        expected_added_meta_graphs=expected_meta_graphs,
        expected_session_logs=expected_session_logs)

  # TODO(ptucker): Test number and contents of checkpoint files.
  def _assert_ckpt(self, output_dir, expected=True):
    ckpt_state = checkpoint_management.get_checkpoint_state(output_dir)
    if expected:
      pattern = '%s/model.ckpt-.*' % output_dir
      primary_ckpt_path = ckpt_state.model_checkpoint_path
      self.assertRegexpMatches(primary_ckpt_path, pattern)
      all_ckpt_paths = ckpt_state.all_model_checkpoint_paths
      self.assertTrue(primary_ckpt_path in all_ckpt_paths)
      for ckpt_path in all_ckpt_paths:
        self.assertRegexpMatches(ckpt_path, pattern)
    else:
      self.assertTrue(ckpt_state is None)

  # TODO(ptucker): Test lock, multi-threaded access?
  def test_summary_writer(self):
    writer = learn.graph_actions.get_summary_writer('log/dir/0')
    self._assert_summaries('log/dir/0', writer)
    self.assertTrue(
        learn.graph_actions.get_summary_writer('log/dir/0') is
        learn.graph_actions.get_summary_writer('log/dir/0'))
    self.assertTrue(
        learn.graph_actions.get_summary_writer('log/dir/0') is
        not learn.graph_actions.get_summary_writer('log/dir/1'))

  # TODO(ptucker): Test restore_checkpoint_path for eval; this should obsolete
  # test_evaluate_with_saver().
  # TODO(ptucker): Test start_queue_runners for both eval & train.
  # TODO(ptucker): Test coord.request_stop & coord.join for eval.

  def _build_inference_graph(self):
    """Build simple inference graph.

    This includes a regular variable, local variable, and fake table.

    Returns:
      Tuple of 3 `Tensor` objects, 2 input and 1 output.
    """
    variables_lib.create_global_step()
    in0 = variables.Variable(1.0)
    in1 = variables_lib.local_variable(2.0)
    fake_table = variables.Variable(
        3.0,
        trainable=False,
        collections=['fake_tables'],
        name='fake_table_var')
    in0.graph.add_to_collections([ops.GraphKeys.TABLE_INITIALIZERS],
                                 fake_table.initializer)
    out = in0 + in1 + fake_table
    return in0, in1, out

  def test_infer(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      self._assert_ckpt(self._output_dir, False)
      in0, in1, out = self._build_inference_graph()
      self.assertEqual({
          'a': 1.0,
          'b': 2.0,
          'c': 6.0
      }, learn.graph_actions.infer(None, {'a': in0,
                                          'b': in1,
                                          'c': out}))
      self._assert_ckpt(self._output_dir, False)

  @test.mock.patch.object(
      learn.graph_actions.coordinator.Coordinator,
      'request_stop',
      side_effect=learn.graph_actions.coordinator.Coordinator.request_stop,
      autospec=True)
  def test_coordinator_request_stop_called(self, request_stop):
    with ops.Graph().as_default() as g, self.test_session(g):
      in0, in1, out = self._build_inference_graph()
      learn.graph_actions.infer(None, {'a': in0, 'b': in1, 'c': out})
      self.assertTrue(request_stop.called)

  @test.mock.patch.object(
      learn.graph_actions.coordinator.Coordinator,
      'request_stop',
      side_effect=learn.graph_actions.coordinator.Coordinator.request_stop,
      autospec=True)
  def test_run_feeds_iter_cleanup_with_exceptions(self, request_stop):
    with ops.Graph().as_default() as g, self.test_session(g):
      in0, in1, out = self._build_inference_graph()
      try:
        for _ in learn.graph_actions.run_feeds_iter({
            'a': in0,
            'b': in1,
            'c': out
        }, [None] * 3):
          self.assertFalse(request_stop.called)
          raise ValueError('Fake exception')
      except ValueError:
        pass
      self.assertTrue(request_stop.called)

  def test_run_feeds_iter_calls_resources_init(self):
    with ops.Graph().as_default():
      in0, _, _ = self._build_inference_graph()
      handle = test_ops.stub_resource_handle_op(container='a', shared_name='b')
      resources.register_resource(
          handle=handle,
          create_op=test_ops.resource_create_op(handle),
          is_initialized_op=test_ops.resource_initialized_op(handle))

      for _ in learn.graph_actions.run_feeds_iter(
          {
              'in0': in0
          }, feed_dicts=[{}]):
        self.assertTrue(test_ops.resource_initialized_op(handle).eval())

  def test_infer_different_default_graph(self):
    with self.test_session():
      self._assert_ckpt(self._output_dir, False)
      with ops.Graph().as_default():
        in0, in1, out = self._build_inference_graph()
      with ops.Graph().as_default():
        self.assertEqual({
            'a': 1.0,
            'b': 2.0,
            'c': 6.0
        }, learn.graph_actions.infer(None, {'a': in0,
                                            'b': in1,
                                            'c': out}))
      self._assert_ckpt(self._output_dir, False)

  def test_infer_invalid_feed(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      self._assert_ckpt(self._output_dir, False)
      in0, _, _ = self._build_inference_graph()
      with self.assertRaisesRegexp(TypeError, 'Can not convert a NoneType'):
        learn.graph_actions.infer(None, {'a': in0}, feed_dict={None: 4.0})
      self._assert_ckpt(self._output_dir, False)

  def test_infer_feed(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      self._assert_ckpt(self._output_dir, False)
      in0, _, out = self._build_inference_graph()
      self.assertEqual(
          {
              'c': 9.0
          },
          learn.graph_actions.infer(
              None, {'c': out}, feed_dict={in0: 4.0}))
      self._assert_ckpt(self._output_dir, False)

  # TODO(ptucker): Test eval for 1 epoch.

  def test_evaluate_invalid_args(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      self._assert_ckpt(self._output_dir, False)
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.evaluate(
            g,
            output_dir=None,
            checkpoint_path=None,
            eval_dict={'a': constant_op.constant(1.0)})
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.evaluate(
            g,
            output_dir='',
            checkpoint_path=None,
            eval_dict={'a': constant_op.constant(1.0)})
      self._assert_ckpt(self._output_dir, False)

  def test_evaluate(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      _, _, out = self._build_inference_graph()
      writer = learn.graph_actions.get_summary_writer(self._output_dir)
      self._assert_summaries(self._output_dir, writer, expected_session_logs=[])
      self._assert_ckpt(self._output_dir, False)
      results = learn.graph_actions.evaluate(
          g,
          output_dir=self._output_dir,
          checkpoint_path=None,
          eval_dict={'a': out},
          max_steps=1)
      self.assertEqual(({'a': 6.0}, 0), results)
      self._assert_summaries(
          self._output_dir,
          writer,
          expected_summaries={0: {
              'a': 6.0
          }},
          expected_session_logs=[])
      self._assert_ckpt(self._output_dir, False)

  def test_evaluate_ready_for_local_init(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      variables_lib.create_global_step()
      v = variables.Variable(1.0)
      variables.Variable(
          v + 1, collections=[ops.GraphKeys.LOCAL_VARIABLES], trainable=False)
      ready_for_local_init_op = variables.report_uninitialized_variables(
          variables.global_variables())
      ops.add_to_collection(ops.GraphKeys.READY_FOR_LOCAL_INIT_OP,
                            ready_for_local_init_op)
      _ = learn.graph_actions.evaluate(
          g,
          output_dir=self._output_dir,
          checkpoint_path=None,
          eval_dict={'a': v},
          max_steps=1)

  def test_evaluate_feed_fn(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      in0, _, out = self._build_inference_graph()
      writer = learn.graph_actions.get_summary_writer(self._output_dir)
      self._assert_summaries(self._output_dir, writer, expected_session_logs=[])
      self._assert_ckpt(self._output_dir, False)
      feeder = _Feeder(in0, 3)
      results = learn.graph_actions.evaluate(
          g,
          output_dir=self._output_dir,
          checkpoint_path=None,
          eval_dict={'a': out},
          feed_fn=feeder.feed_fn,
          max_steps=3)
      self.assertEqual(3, feeder.step)
      self.assertEqual(({'a': 25.0}, 0), results)
      self._assert_summaries(
          self._output_dir,
          writer,
          expected_summaries={0: {
              'a': 25.0
          }},
          expected_session_logs=[])
      self._assert_ckpt(self._output_dir, False)

  def test_evaluate_feed_fn_with_exhaustion(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      in0, _, out = self._build_inference_graph()
      writer = learn.graph_actions.get_summary_writer(self._output_dir)
      self._assert_summaries(self._output_dir, writer, expected_session_logs=[])
      feeder = _Feeder(in0, 2)
      results = learn.graph_actions.evaluate(
          g,
          output_dir=self._output_dir,
          checkpoint_path=None,
          eval_dict={'a': out},
          feed_fn=feeder.feed_fn,
          max_steps=3)
      self.assertEqual(2, feeder.step)
      self.assertEqual(({'a': 15.0}, 0), results)
      self._assert_summaries(
          self._output_dir,
          writer,
          expected_summaries={0: {
              'a': 15.0
          }},
          expected_session_logs=[])

  def test_evaluate_with_saver(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      _, _, out = self._build_inference_graph()
      ops.add_to_collection(ops.GraphKeys.SAVERS, saver_lib.Saver())
      writer = learn.graph_actions.get_summary_writer(self._output_dir)
      self._assert_summaries(self._output_dir, writer, expected_session_logs=[])
      results = learn.graph_actions.evaluate(
          g,
          output_dir=self._output_dir,
          checkpoint_path=None,
          eval_dict={'a': out},
          max_steps=1)
      self.assertEqual(({'a': 6.0}, 0), results)
      self._assert_summaries(
          self._output_dir,
          writer,
          expected_summaries={0: {
              'a': 6.0
          }},
          expected_session_logs=[])

  # TODO(ptucker): Resume training from previous ckpt.
  # TODO(ptucker): !supervisor_is_chief
  # TODO(ptucker): Custom init op for training.
  # TODO(ptucker): Mock supervisor, and assert all interactions.


# TODO(ispir): remove following tests after deprecated train.
class GraphActionsTrainTest(test.TestCase):
  """Tests for train."""

  def setUp(self):
    learn.graph_actions.clear_summary_writers()
    self._output_dir = tempfile.mkdtemp()
    testing.FakeSummaryWriter.install()

  def tearDown(self):
    testing.FakeSummaryWriter.uninstall()
    if self._output_dir:
      shutil.rmtree(self._output_dir)
    learn.graph_actions.clear_summary_writers()

  def _assert_summaries(self,
                        output_dir,
                        expected_summaries=None,
                        expected_graphs=None,
                        expected_meta_graphs=None,
                        expected_session_logs=None):
    writer = learn.graph_actions.get_summary_writer(output_dir)
    self.assertTrue(isinstance(writer, testing.FakeSummaryWriter))
    writer.assert_summaries(
        self,
        expected_logdir=output_dir,
        expected_graph=ops.get_default_graph(),
        expected_summaries=expected_summaries,
        expected_added_graphs=expected_graphs,
        expected_added_meta_graphs=expected_meta_graphs,
        expected_session_logs=expected_session_logs)

  # TODO(ptucker): Test number and contents of checkpoint files.
  def _assert_ckpt(self, output_dir, expected=True):
    ckpt_state = checkpoint_management.get_checkpoint_state(output_dir)
    if expected:
      pattern = '%s/model.ckpt-.*' % output_dir
      primary_ckpt_path = ckpt_state.model_checkpoint_path
      self.assertRegexpMatches(primary_ckpt_path, pattern)
      all_ckpt_paths = ckpt_state.all_model_checkpoint_paths
      self.assertTrue(primary_ckpt_path in all_ckpt_paths)
      for ckpt_path in all_ckpt_paths:
        self.assertRegexpMatches(ckpt_path, pattern)
    else:
      self.assertTrue(ckpt_state is None)

  def _build_inference_graph(self):
    """Build simple inference graph.

    This includes a regular variable, local variable, and fake table.

    Returns:
      Tuple of 3 `Tensor` objects, 2 input and 1 output.
    """
    variables_lib.create_global_step()
    in0 = variables.Variable(1.0)
    in1 = variables_lib.local_variable(2.0)
    fake_table = variables.Variable(
        3.0,
        trainable=False,
        collections=['fake_tables'],
        name='fake_table_var')
    in0.graph.add_to_collections([ops.GraphKeys.TABLE_INITIALIZERS],
                                 fake_table.initializer)
    out = in0 + in1 + fake_table
    return in0, in1, out

  def test_train_invalid_args(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      train_op = constant_op.constant(1.0)
      loss_op = constant_op.constant(2.0)
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.train(
            g, output_dir=None, train_op=train_op, loss_op=loss_op)
      with self.assertRaisesRegexp(ValueError, 'utput directory'):
        learn.graph_actions.train(
            g,
            output_dir='',
            train_op=constant_op.constant(1.0),
            loss_op=constant_op.constant(2.0))
      with self.assertRaisesRegexp(ValueError, 'train_op'):
        learn.graph_actions.train(
            g, output_dir=self._output_dir, train_op=None, loss_op=loss_op)
      with self.assertRaisesRegexp(ValueError, 'loss_op'):
        learn.graph_actions.train(
            g,
            output_dir=self._output_dir,
            train_op=constant_op.constant(1.0),
            loss_op=None)
      with self.assertRaisesRegexp(ValueError, 'global_step'):
        learn.graph_actions.train(
            g,
            output_dir=self._output_dir,
            train_op=constant_op.constant(1.0),
            loss_op=loss_op)

  # TODO(ptucker): Resume training from previous ckpt.
  # TODO(ptucker): !supervisor_is_chief
  # TODO(ptucker): Custom init op for training.
  # TODO(ptucker): Mock supervisor, and assert all interactions.

  def test_train(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      self._assert_summaries(self._output_dir)
      self._assert_ckpt(self._output_dir, False)
      loss = learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=constant_op.constant(2.0),
          steps=1)
      # TODO(ebrevdo,ptucker,ispir): this meta_graph_def lacks the
      # SaverDef, so we can't add it to the summary assertion test below.
      # meta_graph_def = meta_graph.create_meta_graph_def()
      self.assertEqual(2.0, loss)
      self._assert_summaries(self._output_dir, expected_graphs=[g])
      self._assert_ckpt(self._output_dir, True)

  def test_train_steps_is_incremental(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=constant_op.constant(2.0),
          steps=10)
      step = checkpoint_utils.load_variable(
          self._output_dir, variables_lib.get_global_step().name)
      self.assertEqual(10, step)

    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=constant_op.constant(2.0),
          steps=15)
      step = checkpoint_utils.load_variable(
          self._output_dir, variables_lib.get_global_step().name)
      self.assertEqual(25, step)

  def test_train_max_steps_is_not_incremental(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=constant_op.constant(2.0),
          max_steps=10)
      step = checkpoint_utils.load_variable(
          self._output_dir, variables_lib.get_global_step().name)
      self.assertEqual(10, step)

    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=constant_op.constant(2.0),
          max_steps=15)
      step = checkpoint_utils.load_variable(
          self._output_dir, variables_lib.get_global_step().name)
      self.assertEqual(15, step)

  def test_train_loss(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      variables_lib.create_global_step()
      loss_var = variables_lib.local_variable(10.0)
      train_op = control_flow_ops.group(
          state_ops.assign_add(variables_lib.get_global_step(), 1),
          state_ops.assign_add(loss_var, -1.0))
      self._assert_summaries(self._output_dir)
      self._assert_ckpt(self._output_dir, False)
      loss = learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=loss_var.value(),
          steps=6)
      # TODO(ebrevdo,ptucker,ispir): this meta_graph_def lacks the
      # SaverDef, so we can't add it to the summary assertion test below.
      # meta_graph_def = meta_graph.create_meta_graph_def()
      self.assertEqual(4.0, loss)
      self._assert_summaries(self._output_dir, expected_graphs=[g])
      self._assert_ckpt(self._output_dir, True)

  def test_train_summaries(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      loss_op = constant_op.constant(2.0)
      summary.scalar('loss', loss_op)
      self._assert_summaries(self._output_dir)
      self._assert_ckpt(self._output_dir, False)
      loss = learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=loss_op,
          steps=1)
      # TODO(ebrevdo,ptucker,ispir): this meta_graph_def lacks the
      # SaverDef, so we can't add it to the summary assertion test below.
      # meta_graph_def = meta_graph.create_meta_graph_def()
      self.assertEqual(2.0, loss)
      self._assert_summaries(
          self._output_dir,
          expected_graphs=[g],
          expected_summaries={1: {
              'loss': 2.0
          }})
      self._assert_ckpt(self._output_dir, True)

  def test_train_chief_monitor(self):
    with ops.Graph().as_default() as g, self.test_session(g):
      with ops.control_dependencies(self._build_inference_graph()):
        train_op = state_ops.assign_add(variables_lib.get_global_step(), 1)
      loss_op = constant_op.constant(2.0)
      summary.scalar('loss', loss_op)
      chief_exclusive_monitor = _BaseMonitorWrapper(False)
      all_workers_monitor = _BaseMonitorWrapper(True)
      loss = learn.graph_actions.train(
          g,
          output_dir=self._output_dir,
          train_op=train_op,
          loss_op=loss_op,
          supervisor_is_chief=True,
          steps=1,
          monitors=[chief_exclusive_monitor, all_workers_monitor])
      self.assertEqual(2.0, loss)
      self.assertTrue(chief_exclusive_monitor.is_active and
                      all_workers_monitor.is_active,
                      'All monitors must have been active.')
      self.assertTrue(chief_exclusive_monitor.has_step and
                      all_workers_monitor.has_step,
                      'All monitors must have a step.')

  def test_train_worker_monitor(self):
    # We need to explicitly set device due to check on non-chief workers
    # requiring all variables to have a device assigned.
    with ops.Graph().as_default() as g, g.device('/cpu:0'):
      global_step = variables_lib.create_global_step(g)
      train_op = state_ops.assign_add(global_step, 1)
      loss_op = constant_op.constant(2.0)
      summary.scalar('loss', loss_op)
      # Add explicit "local" init op to initialize all variables
      # as there's no chief to init here.
      init_op = variables.global_variables_initializer()
      ops.add_to_collection(ops.GraphKeys.LOCAL_INIT_OP, init_op)
      # Create worker monitors where one should be active on the worker
      # and the other chief exclusive.
      chief_exclusive_monitor = _BaseMonitorWrapper(False)
      all_workers_monitor = _BaseMonitorWrapper(True)
      with self.test_session(g):
        loss = learn.graph_actions.train(
            g,
            output_dir=self._output_dir,
            global_step_tensor=global_step,
            train_op=train_op,
            loss_op=loss_op,
            supervisor_is_chief=False,
            steps=1,
            monitors=[chief_exclusive_monitor, all_workers_monitor])
      self.assertEqual(2.0, loss)
      self.assertTrue(not chief_exclusive_monitor.is_active and
                      all_workers_monitor.is_active,
                      'Only non-chief runnable monitor must have been active.')
      self.assertTrue(not chief_exclusive_monitor.has_step and
                      all_workers_monitor.has_step,
                      'Only non-chief runnable monitor must have a step.')


if __name__ == '__main__':
  test.main()
