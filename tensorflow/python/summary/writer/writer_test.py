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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.util import compat


class SummaryWriterTestCase(test.TestCase):

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
    return summary_iterator.summary_iterator(event_paths[-1])

  def _assertRecent(self, t):
    self.assertTrue(abs(t - time.time()) < 5)

  def _assertEventsWithGraph(self, test_dir, g, has_shapes):
    meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=g.as_graph_def(add_shapes=has_shapes))

    rr = self._EventsReader(test_dir)

    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:2", ev.file_version)

    # The next event should have the graph.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(0, ev.step)
    ev_graph = graph_pb2.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(g.as_graph_def(add_shapes=has_shapes), ev_graph)

    # The next event should have the metagraph.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(0, ev.step)
    ev_meta_graph = meta_graph_pb2.MetaGraphDef()
    ev_meta_graph.ParseFromString(ev.meta_graph_def)
    self.assertProtoEquals(meta_graph_def, ev_meta_graph)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testAddingSummaryGraphAndRunMetadata(self):
    test_dir = self._CleanTestDir("basics")
    sw = writer.FileWriter(test_dir)

    sw.add_session_log(event_pb2.SessionLog(status=SessionLog.START), 1)
    sw.add_summary(
        summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(
                tag="mee", simple_value=10.0)]),
        10)
    sw.add_summary(
        summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(
                tag="boo", simple_value=20.0)]),
        20)
    with ops.Graph().as_default() as g:
      constant_op.constant([0], name="zero")
    sw.add_graph(g, global_step=30)

    run_metadata = config_pb2.RunMetadata()
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
    ev_graph = graph_pb2.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(g.as_graph_def(add_shapes=True), ev_graph)

    # The next event should have metadata for the run.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(40, ev.step)
    self.assertEquals("test run", ev.tagged_run_metadata.tag)
    parsed_run_metadata = config_pb2.RunMetadata()
    parsed_run_metadata.ParseFromString(ev.tagged_run_metadata.run_metadata)
    self.assertProtoEquals(run_metadata, parsed_run_metadata)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testGraphAsNamed(self):
    test_dir = self._CleanTestDir("basics_named_graph")
    with ops.Graph().as_default() as g:
      constant_op.constant([12], name="douze")
    sw = writer.FileWriter(test_dir, graph=g)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, True)

  def testGraphAsPositional(self):
    test_dir = self._CleanTestDir("basics_positional_graph")
    with ops.Graph().as_default() as g:
      constant_op.constant([12], name="douze")
    sw = writer.FileWriter(test_dir, g)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, True)

  def testGraphDefAsNamed(self):
    test_dir = self._CleanTestDir("basics_named_graph_def")
    with ops.Graph().as_default() as g:
      constant_op.constant([12], name="douze")
    gd = g.as_graph_def()
    sw = writer.FileWriter(test_dir, graph_def=gd)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, False)

  def testGraphDefAsPositional(self):
    test_dir = self._CleanTestDir("basics_positional_graph_def")
    with ops.Graph().as_default() as g:
      constant_op.constant([12], name="douze")
    gd = g.as_graph_def()
    sw = writer.FileWriter(test_dir, gd)
    sw.close()
    self._assertEventsWithGraph(test_dir, g, False)

  def testGraphAndGraphDef(self):
    with self.assertRaises(ValueError):
      test_dir = self._CleanTestDir("basics_graph_and_graph_def")
      with ops.Graph().as_default() as g:
        constant_op.constant([12], name="douze")
      gd = g.as_graph_def()
      sw = writer.FileWriter(test_dir, graph=g, graph_def=gd)
      sw.close()

  def testNeitherGraphNorGraphDef(self):
    with self.assertRaises(TypeError):
      test_dir = self._CleanTestDir("basics_string_instead_of_graph")
      sw = writer.FileWriter(test_dir, "string instead of graph object")
      sw.close()

  def testCloseAndReopen(self):
    test_dir = self._CleanTestDir("close_and_reopen")
    sw = writer.FileWriter(test_dir)
    sw.add_session_log(event_pb2.SessionLog(status=SessionLog.START), 1)
    sw.close()
    # Sleep at least one second to make sure we get a new event file name.
    time.sleep(1.2)
    sw.reopen()
    sw.add_session_log(event_pb2.SessionLog(status=SessionLog.START), 2)
    sw.close()

    # We should now have 2 events files.
    event_paths = sorted(glob.glob(os.path.join(test_dir, "event*")))
    self.assertEquals(2, len(event_paths))

    # Check the first file contents.
    rr = summary_iterator.summary_iterator(event_paths[0])
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
    rr = summary_iterator.summary_iterator(event_paths[1])
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

  def testNonBlockingClose(self):
    test_dir = self._CleanTestDir("non_blocking_close")
    sw = writer.FileWriter(test_dir)
    # Sleep 1.2 seconds to make sure event queue is empty.
    time.sleep(1.2)
    time_before_close = time.time()
    sw.close()
    self._assertRecent(time_before_close)

  def testWithStatement(self):
    test_dir = self._CleanTestDir("with_statement")
    with writer.FileWriter(test_dir) as sw:
      sw.add_session_log(event_pb2.SessionLog(status=SessionLog.START), 1)
    event_paths = sorted(glob.glob(os.path.join(test_dir, "event*")))
    self.assertEquals(1, len(event_paths))

  # Checks that values returned from session Run() calls are added correctly to
  # summaries.  These are numpy types so we need to check they fit in the
  # protocol buffers correctly.
  def testAddingSummariesFromSessionRunCalls(self):
    test_dir = self._CleanTestDir("global_step")
    sw = writer.FileWriter(test_dir)
    with self.test_session():
      i = constant_op.constant(1, dtype=dtypes.int32, shape=[])
      l = constant_op.constant(2, dtype=dtypes.int64, shape=[])
      # Test the summary can be passed serialized.
      summ = summary_pb2.Summary(
          value=[summary_pb2.Summary.Value(
              tag="i", simple_value=1.0)])
      sw.add_summary(summ.SerializeToString(), i.eval())
      sw.add_summary(
          summary_pb2.Summary(
              value=[summary_pb2.Summary.Value(
                  tag="l", simple_value=2.0)]),
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

  def testPluginMetadataStrippedFromSubsequentEvents(self):
    test_dir = self._CleanTestDir("basics")
    sw = writer.FileWriter(test_dir)

    sw.add_session_log(event_pb2.SessionLog(status=SessionLog.START), 1)

    # We add 2 summaries with the same tags. They both have metadata. The writer
    # should strip the metadata from the second one.
    value = summary_pb2.Summary.Value(tag="foo", simple_value=10.0)
    value.metadata.plugin_data.plugin_name = "bar"
    value.metadata.plugin_data.content = compat.as_bytes("... content ...")
    sw.add_summary(summary_pb2.Summary(value=[value]), 10)
    value = summary_pb2.Summary.Value(tag="foo", simple_value=10.0)
    value.metadata.plugin_data.plugin_name = "bar"
    value.metadata.plugin_data.content = compat.as_bytes("... content ...")
    sw.add_summary(summary_pb2.Summary(value=[value]), 10)

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

    # This is the first event with tag foo. It should contain SummaryMetadata.
    ev = next(rr)
    self.assertProtoEquals("""
      value {
        tag: "foo"
        simple_value: 10.0
        metadata {
          plugin_data {
            plugin_name: "bar"
            content: "... content ..."
          }
        }
      }
      """, ev.summary)

    # This is the second event with tag foo. It should lack SummaryMetadata
    # because the file writer should have stripped it.
    ev = next(rr)
    self.assertProtoEquals("""
      value {
        tag: "foo"
        simple_value: 10.0
      }
      """, ev.summary)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testFileWriterWithSuffix(self):
    test_dir = self._CleanTestDir("test_suffix")
    sw = writer.FileWriter(test_dir, filename_suffix="_test_suffix")
    for _ in range(10):
      sw.add_summary(
          summary_pb2.Summary(value=[
              summary_pb2.Summary.Value(tag="float_ten", simple_value=10.0)
          ]),
          10)
      sw.close()
      sw.reopen()
    sw.close()
    event_filenames = glob.glob(os.path.join(test_dir, "event*"))
    for filename in event_filenames:
      self.assertTrue(filename.endswith("_test_suffix"))


class SummaryWriterCacheTest(test.TestCase):
  """SummaryWriterCache tests."""

  def _test_dir(self, test_name):
    """Create an empty dir to use for tests.

    Args:
      test_name: Name of the test.

    Returns:
      Absolute path to the test directory.
    """
    test_dir = os.path.join(self.get_temp_dir(), test_name)
    if os.path.isdir(test_dir):
      for f in glob.glob("%s/*" % test_dir):
        os.remove(f)
    else:
      os.makedirs(test_dir)
    return test_dir

  def test_cache(self):
    with ops.Graph().as_default():
      dir1 = self._test_dir("test_cache_1")
      dir2 = self._test_dir("test_cache_2")
      sw1 = writer_cache.FileWriterCache.get(dir1)
      sw2 = writer_cache.FileWriterCache.get(dir2)
      sw3 = writer_cache.FileWriterCache.get(dir1)
      self.assertEqual(sw1, sw3)
      self.assertFalse(sw1 == sw2)
      sw1.close()
      sw2.close()
      events1 = glob.glob(os.path.join(dir1, "event*"))
      self.assertTrue(events1)
      events2 = glob.glob(os.path.join(dir2, "event*"))
      self.assertTrue(events2)
      events3 = glob.glob(os.path.join("nowriter", "event*"))
      self.assertFalse(events3)

  def test_clear(self):
    with ops.Graph().as_default():
      dir1 = self._test_dir("test_clear")
      sw1 = writer_cache.FileWriterCache.get(dir1)
      writer_cache.FileWriterCache.clear()
      sw2 = writer_cache.FileWriterCache.get(dir1)
      self.assertFalse(sw1 == sw2)


class ExamplePluginAsset(plugin_asset.PluginAsset):
  plugin_name = "example"

  def assets(self):
    return {"foo.txt": "foo!", "bar.txt": "bar!"}


class PluginAssetsTest(test.TestCase):

  def testPluginAssetSerialized(self):
    with ops.Graph().as_default() as g:
      plugin_asset.get_plugin_asset(ExamplePluginAsset)

      logdir = self.get_temp_dir()
      fw = writer.FileWriter(logdir)
      fw.add_graph(g)
    plugin_dir = os.path.join(logdir, writer._PLUGINS_DIR, "example")

    with gfile.Open(os.path.join(plugin_dir, "foo.txt"), "r") as f:
      content = f.read()
    self.assertEqual(content, "foo!")

    with gfile.Open(os.path.join(plugin_dir, "bar.txt"), "r") as f:
      content = f.read()
    self.assertEqual(content, "bar!")


if __name__ == "__main__":
  test.main()
