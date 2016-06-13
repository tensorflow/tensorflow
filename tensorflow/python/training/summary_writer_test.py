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

"""Tests for training_coordinator.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path
import shutil
import time

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog


class SummaryWriterTestCase(tf.test.TestCase):

  def _TestDir(self, test_name):
    test_dir = os.path.join(self.get_temp_dir(), test_name)
    return test_dir

  def _CleanTestDir(self, test_name):
    test_dir = self._TestDir(test_name)
    if os.path.exists(test_dir):
      shutil.rmtree(test_dir)
    return test_dir

  def _EventsReader(self, test_dir):
    event_paths = glob.glob(os.path.join(test_dir, "event*"))
    # If the tests runs multiple times in the same directory we can have
    # more than one matching event file.  We only want to read the last one.
    self.assertTrue(event_paths)
    return tf.train.summary_iterator(event_paths[-1])

  def _assertRecent(self, t):
    self.assertTrue(abs(t - time.time()) < 5)

  def _assertEventsWithGraph(self, test_dir, g, has_shapes):
    rr = self._EventsReader(test_dir)

    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:2", ev.file_version)

    # The next event should have the graph.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(0, ev.step)
    ev_graph = tf.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(g.as_graph_def(add_shapes=has_shapes), ev_graph)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testAddingSummaryGraphAndRunMetadata(self):
    test_dir = self._CleanTestDir("basics")
    sw = tf.train.SummaryWriter(test_dir)

    sw.add_session_log(tf.SessionLog(status=SessionLog.START), 1)
    sw.add_summary(tf.Summary(value=[tf.Summary.Value(tag="mee",
                                                      simple_value=10.0)]),
                   10)
    sw.add_summary(tf.Summary(value=[tf.Summary.Value(tag="boo",
                                                      simple_value=20.0)]),
                   20)
    with tf.Graph().as_default() as g:
      tf.constant([0], name="zero")
    sw.add_graph(g, global_step=30)

    run_metadata = tf.RunMetadata()
    device_stats = run_metadata.step_stats.dev_stats.add()
    device_stats.device = "test"
    sw.add_run_metadata(run_metadata, "test run", global_step=40)
    sw.close()
    rr = self._EventsReader(test_dir)

    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:2", ev.file_version)

    # The next event should be the START message.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(1, ev.step)
    self.assertEquals(SessionLog.START, ev.session_log.status)

    # The next event should have the value 'mee=10.0'.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(10, ev.step)
    self.assertProtoEquals("""
      value { tag: 'mee' simple_value: 10.0 }
      """, ev.summary)

    # The next event should have the value 'boo=20.0'.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(20, ev.step)
    self.assertProtoEquals("""
      value { tag: 'boo' simple_value: 20.0 }
      """, ev.summary)

    # The next event should have the graph_def.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(30, ev.step)
    ev_graph = tf.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(g.as_graph_def(add_shapes=True), ev_graph)

    # The next event should have metadata for the run.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(40, ev.step)
    self.assertEquals("test run", ev.tagged_run_metadata.tag)
    parsed_run_metadata = tf.RunMetadata()
    parsed_run_metadata.ParseFromString(ev.tagged_run_metadata.run_metadata)
    self.assertProtoEquals(run_metadata, parsed_run_metadata)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testGraphAsNamed(self):
    test_dir = self._CleanTestDir("basics_named_graph")
    with tf.Graph().as_default() as g:
      tf.constant([12], name="douze")
    sw = tf.train.SummaryWriter(test_dir, graph=g)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, True)

  def testGraphAsPositional(self):
    test_dir = self._CleanTestDir("basics_positional_graph")
    with tf.Graph().as_default() as g:
      tf.constant([12], name="douze")
    sw = tf.train.SummaryWriter(test_dir, g)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, True)

  def testGraphDefAsNamed(self):
    test_dir = self._CleanTestDir("basics_named_graph_def")
    with tf.Graph().as_default() as g:
      tf.constant([12], name="douze")
    gd = g.as_graph_def()
    sw = tf.train.SummaryWriter(test_dir, graph_def=gd)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, False)

  def testGraphDefAsPositional(self):
    test_dir = self._CleanTestDir("basics_positional_graph_def")
    with tf.Graph().as_default() as g:
      tf.constant([12], name="douze")
    gd = g.as_graph_def()
    sw = tf.train.SummaryWriter(test_dir, gd)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, False)

  def testGraphAndGraphDef(self):
    with self.assertRaises(ValueError):
      test_dir = self._CleanTestDir("basics_graph_and_graph_def")
      with tf.Graph().as_default() as g:
        tf.constant([12], name="douze")
      gd = g.as_graph_def()
      sw = tf.train.SummaryWriter(test_dir, graph=g, graph_def=gd)
      sw.close()

  def testNeitherGraphNorGraphDef(self):
    with self.assertRaises(TypeError):
      test_dir = self._CleanTestDir("basics_string_instead_of_graph")
      sw = tf.train.SummaryWriter(test_dir, "string instead of graph object")
      sw.close()

  def testCloseAndReopen(self):
    test_dir = self._CleanTestDir("close_and_reopen")
    sw = tf.train.SummaryWriter(test_dir)
    sw.add_session_log(tf.SessionLog(status=SessionLog.START), 1)
    sw.close()
    # Sleep at least one second to make sure we get a new event file name.
    time.sleep(1.2)
    sw.reopen()
    sw.add_session_log(tf.SessionLog(status=SessionLog.START), 2)
    sw.close()

    # We should now have 2 events files.
    event_paths = sorted(glob.glob(os.path.join(test_dir, "event*")))
    self.assertEquals(2, len(event_paths))

    # Check the first file contents.
    rr = tf.train.summary_iterator(event_paths[0])
    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:2", ev.file_version)
    # The next event should be the START message.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(1, ev.step)
    self.assertEquals(SessionLog.START, ev.session_log.status)
    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

    # Check the second file contents.
    rr = tf.train.summary_iterator(event_paths[1])
    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:2", ev.file_version)
    # The next event should be the START message.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(2, ev.step)
    self.assertEquals(SessionLog.START, ev.session_log.status)
    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  # Checks that values returned from session Run() calls are added correctly to
  # summaries.  These are numpy types so we need to check they fit in the
  # protocol buffers correctly.
  def testAddingSummariesFromSessionRunCalls(self):
    test_dir = self._CleanTestDir("global_step")
    sw = tf.train.SummaryWriter(test_dir)
    with self.test_session():
      i = tf.constant(1, dtype=tf.int32, shape=[])
      l = tf.constant(2, dtype=tf.int64, shape=[])
      # Test the summary can be passed serialized.
      summ = tf.Summary(value=[tf.Summary.Value(tag="i", simple_value=1.0)])
      sw.add_summary(summ.SerializeToString(), i.eval())
      sw.add_summary(tf.Summary(value=[tf.Summary.Value(tag="l",
                                                        simple_value=2.0)]),
                     l.eval())
      sw.close()

    rr = self._EventsReader(test_dir)

    # File_version.
    ev = next(rr)
    self.assertTrue(ev)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:2", ev.file_version)

    # Summary passed serialized.
    ev = next(rr)
    self.assertTrue(ev)
    self._assertRecent(ev.wall_time)
    self.assertEquals(1, ev.step)
    self.assertProtoEquals("""
      value { tag: 'i' simple_value: 1.0 }
      """, ev.summary)

    # Summary passed as SummaryObject.
    ev = next(rr)
    self.assertTrue(ev)
    self._assertRecent(ev.wall_time)
    self.assertEquals(2, ev.step)
    self.assertProtoEquals("""
      value { tag: 'l' simple_value: 2.0 }
      """, ev.summary)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))


if __name__ == "__main__":
  tf.test.main()
