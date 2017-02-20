# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Fake summary writer for unit tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache


# TODO(ptucker): Replace with mock framework.
class FakeSummaryWriter(object):
  """Fake summary writer."""

  _replaced_summary_writer = None

  @classmethod
  def install(cls):
    if cls._replaced_summary_writer:
      raise ValueError('FakeSummaryWriter already installed.')
    cls._replaced_summary_writer = writer.FileWriter
    writer.FileWriter = FakeSummaryWriter
    writer_cache.FileWriter = FakeSummaryWriter

  @classmethod
  def uninstall(cls):
    if not cls._replaced_summary_writer:
      raise ValueError('FakeSummaryWriter not installed.')
    writer.FileWriter = cls._replaced_summary_writer
    writer_cache.FileWriter = cls._replaced_summary_writer
    cls._replaced_summary_writer = None

  def __init__(self, logdir, graph=None):
    self._logdir = logdir
    self._graph = graph
    self._summaries = {}
    self._added_graphs = []
    self._added_meta_graphs = []
    self._added_session_logs = []

  @property
  def summaries(self):
    return self._summaries

  def assert_summaries(self,
                       test_case,
                       expected_logdir=None,
                       expected_graph=None,
                       expected_summaries=None,
                       expected_added_graphs=None,
                       expected_added_meta_graphs=None,
                       expected_session_logs=None):
    """Assert expected items have been added to summary writer."""
    if expected_logdir is not None:
      test_case.assertEqual(expected_logdir, self._logdir)
    if expected_graph is not None:
      test_case.assertTrue(expected_graph is self._graph)
    expected_summaries = expected_summaries or {}
    for step in expected_summaries:
      test_case.assertTrue(
          step in self._summaries,
          msg='Missing step %s from %s.' % (step, self._summaries.keys()))
      actual_simple_values = {}
      for step_summary in self._summaries[step]:
        for v in step_summary.value:
          # Ignore global_step/sec since it's written by Supervisor in a
          # separate thread, so it's non-deterministic how many get written.
          if 'global_step/sec' != v.tag:
            actual_simple_values[v.tag] = v.simple_value
      test_case.assertEqual(expected_summaries[step], actual_simple_values)
    if expected_added_graphs is not None:
      test_case.assertEqual(expected_added_graphs, self._added_graphs)
    if expected_added_meta_graphs is not None:
      test_case.assertEqual(expected_added_meta_graphs, self._added_meta_graphs)
    if expected_session_logs is not None:
      test_case.assertEqual(expected_session_logs, self._added_session_logs)

  def add_summary(self, summ, current_global_step):
    """Add summary."""
    if isinstance(summ, bytes):
      summary_proto = summary_pb2.Summary()
      summary_proto.ParseFromString(summ)
      summ = summary_proto
    if current_global_step in self._summaries:
      step_summaries = self._summaries[current_global_step]
    else:
      step_summaries = []
      self._summaries[current_global_step] = step_summaries
    step_summaries.append(summ)

  # NOTE: Ignore global_step since its value is non-deterministic.
  def add_graph(self, graph, global_step=None, graph_def=None):
    """Add graph."""
    if (global_step is not None) and (global_step < 0):
      raise ValueError('Invalid global_step %s.' % global_step)
    if graph_def is not None:
      raise ValueError('Unexpected graph_def %s.' % graph_def)
    self._added_graphs.append(graph)

  def add_meta_graph(self, meta_graph_def, global_step=None):
    """Add metagraph."""
    if (global_step is not None) and (global_step < 0):
      raise ValueError('Invalid global_step %s.' % global_step)
    self._added_meta_graphs.append(meta_graph_def)

  # NOTE: Ignore global_step since its value is non-deterministic.
  def add_session_log(self, session_log, global_step=None):
    # pylint: disable=unused-argument
    self._added_session_logs.append(session_log)

  def flush(self):
    pass

  def reopen(self):
    pass
