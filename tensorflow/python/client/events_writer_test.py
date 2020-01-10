"""Tests for the SWIG-wrapped events writer."""
import os.path

from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class PywrapeventsWriterTest(test_util.TensorFlowTestCase):

  def testWriteEvents(self):
    file_prefix = os.path.join(self.get_temp_dir(), "events")
    writer = pywrap_tensorflow.EventsWriter(file_prefix)
    filename = writer.FileName()
    event_written = event_pb2.Event(
        wall_time=123.45, step=67,
        summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag="foo", simple_value=89.0)]))
    writer.WriteEvent(event_written)
    writer.Flush()
    writer.Close()

    with self.assertRaises(IOError):
      for r in tf_record.tf_record_iterator(filename + "DOES_NOT_EXIST"):
        self.assertTrue(False)

    reader = tf_record.tf_record_iterator(filename)
    event_read = event_pb2.Event()

    event_read.ParseFromString(next(reader))
    self.assertTrue(event_read.HasField("file_version"))

    event_read.ParseFromString(next(reader))
    # Second event
    self.assertProtoEquals("""
    wall_time: 123.45 step: 67
    summary { value { tag: 'foo' simple_value: 89.0 } }
    """, event_read)

    with self.assertRaises(StopIteration):
      next(reader)

  def testWriteEventInvalidType(self):
    class _Invalid(object):
      def __str__(self): return "Invalid"
    with self.assertRaisesRegexp(TypeError, "Invalid"):
      pywrap_tensorflow.EventsWriter("foo").WriteEvent(_Invalid())


if __name__ == "__main__":
  googletest.main()
