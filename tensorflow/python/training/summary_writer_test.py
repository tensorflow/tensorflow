"""Tests for training_coordinator.py."""
import glob
import os.path
import shutil
import time

import tensorflow.python.platform

import tensorflow as tf


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
    # If the tests runs multiple time in the same directory we can have
    # more than one matching event file.  We only want to read the last one.
    self.assertTrue(event_paths)
    return tf.train.summary_iterator(event_paths[-1])

  def _assertRecent(self, t):
    self.assertTrue(abs(t - time.time()) < 5)

  def testBasics(self):
    test_dir = self._CleanTestDir("basics")
    sw = tf.train.SummaryWriter(test_dir)
    sw.add_summary(tf.Summary(value=[tf.Summary.Value(tag="mee",
                                                      simple_value=10.0)]),
                   10)
    sw.add_summary(tf.Summary(value=[tf.Summary.Value(tag="boo",
                                                      simple_value=20.0)]),
                   20)
    with tf.Graph().as_default() as g:
      tf.constant([0], name="zero")
    gd = g.as_graph_def()
    sw.add_graph(gd, global_step=30)
    sw.close()
    rr = self._EventsReader(test_dir)

    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:1", ev.file_version)

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
    self.assertProtoEquals(gd, ev.graph_def)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testConstructWithGraph(self):
    test_dir = self._CleanTestDir("basics_with_graph")
    with tf.Graph().as_default() as g:
      tf.constant([12], name="douze")
    gd = g.as_graph_def()
    sw = tf.train.SummaryWriter(test_dir, graph_def=gd)
    sw.close()
    rr = self._EventsReader(test_dir)

    # The first event should list the file_version.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals("brain.Event:1", ev.file_version)

    # The next event should have the graph.
    ev = next(rr)
    self._assertRecent(ev.wall_time)
    self.assertEquals(0, ev.step)
    self.assertProtoEquals(gd, ev.graph_def)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  # Checks that values returned from session Run() calls are added correctly to
  # summaries.  These are numpy types so we need to check they fit in the
  # protocol buffers correctly.
  def testSummariesAndStopFromSessionRunCalls(self):
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
    self.assertEquals("brain.Event:1", ev.file_version)

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
