# -*- coding: utf-8 -*-
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
"""Integration tests for the Graphs Plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.graphs import graphs_plugin


class GraphsPluginTest(tf.test.TestCase):

  _RUN_WITH_GRAPH = '_RUN_WITH_GRAPH'
  _RUN_WITHOUT_GRAPH = '_RUN_WITHOUT_GRAPH'

  _METADATA_TAG = 'secret-stats'
  _MESSAGE_PREFIX_LENGTH_LOWER_BOUND = 1024

  def generate_run(self, run_name, include_graph):
    """Create a run with a text summary, metadata, and optionally a graph."""
    tf.reset_default_graph()
    k1 = tf.constant(math.pi, name='k1')
    k2 = tf.constant(math.e, name='k2')
    result = (k1 ** k2) - k1
    expected = tf.constant(20.0, name='expected')
    error = tf.abs(result - expected, name='error')
    message_prefix_value = 'error ' * 1000
    true_length = len(message_prefix_value)
    assert true_length > self._MESSAGE_PREFIX_LENGTH_LOWER_BOUND, true_length
    message_prefix = tf.constant(message_prefix_value, name='message_prefix')
    error_message = tf.string_join([message_prefix,
                                    tf.as_string(error, name='error_string')],
                                   name='error_message')
    summary_message = tf.summary.text('summary_message', error_message)

    sess = tf.Session()
    writer = tf.summary.FileWriter(os.path.join(self.logdir, run_name))
    if include_graph:
      writer.add_graph(sess.graph)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    s = sess.run(summary_message, options=options, run_metadata=run_metadata)
    writer.add_summary(s)
    writer.add_run_metadata(run_metadata, self._METADATA_TAG)
    writer.close()

  def set_up_with_runs(self, with_graph=True, without_graph=True):
    self.logdir = self.get_temp_dir()
    if with_graph:
      self.generate_run(self._RUN_WITH_GRAPH, include_graph=True)
    if without_graph:
      self.generate_run(self._RUN_WITHOUT_GRAPH, include_graph=False)
    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    self.plugin = graphs_plugin.GraphsPlugin()
    self.plugin.get_plugin_apps(multiplexer, None)

  def test_index(self):
    self.set_up_with_runs()
    self.assertItemsEqual([self._RUN_WITH_GRAPH], self.plugin.index_impl())

  def test_run_metadata_index(self):
    self.set_up_with_runs()
    self.assertDictEqual({
        self._RUN_WITH_GRAPH: [self._METADATA_TAG],
        self._RUN_WITHOUT_GRAPH: [self._METADATA_TAG],
    }, self.plugin.run_metadata_index_impl())

  def _get_graph(self, *args, **kwargs):
    """Set up runs, then fetch and return the graph as a proto."""
    self.set_up_with_runs()
    (graph_pbtxt, mime_type) = self.plugin.graph_impl(
        self._RUN_WITH_GRAPH, *args, **kwargs)
    self.assertEqual(mime_type, 'text/x-protobuf')
    return text_format.Parse(graph_pbtxt, tf.GraphDef())

  def test_graph_simple(self):
    graph = self._get_graph()
    node_names = set(node.name for node in graph.node)
    self.assertEqual({'k1', 'k2', 'pow', 'sub', 'expected', 'sub_1', 'error',
                      'message_prefix', 'error_string', 'error_message',
                      'summary_message'},
                     node_names)

  def test_graph_large_attrs(self):
    key = 'o---;;-;'
    graph = self._get_graph(
        limit_attr_size=self._MESSAGE_PREFIX_LENGTH_LOWER_BOUND,
        large_attrs_key=key)
    large_attrs = {
        node.name: list(node.attr[key].list.s)
        for node in graph.node
        if key in node.attr
    }
    self.assertEqual({'message_prefix': [b'value']},
                     large_attrs)

  def test_run_metadata(self):
    self.set_up_with_runs()
    (metadata_pbtxt, mime_type) = self.plugin.run_metadata_impl(
        self._RUN_WITH_GRAPH, self._METADATA_TAG)
    self.assertEqual(mime_type, 'text/x-protobuf')
    text_format.Parse(metadata_pbtxt, tf.RunMetadata())
    # If it parses, we're happy.

  def test_is_active_with_graph(self):
    self.set_up_with_runs(with_graph=True, without_graph=False)
    self.assertTrue(self.plugin.is_active())

  def test_is_active_without_graph(self):
    self.set_up_with_runs(with_graph=False, without_graph=True)
    self.assertFalse(self.plugin.is_active())

  def test_is_active_with_both(self):
    self.set_up_with_runs(with_graph=True, without_graph=True)
    self.assertTrue(self.plugin.is_active())


if __name__ == '__main__':
  tf.test.main()
