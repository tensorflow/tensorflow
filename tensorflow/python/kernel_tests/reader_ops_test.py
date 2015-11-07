"""Tests for Reader ops from io_ops."""

import os
import tensorflow.python.platform

import tensorflow as tf


class IdentityReaderTest(tf.test.TestCase):

  def _ExpectRead(self, sess, key, value, expected):
    k, v = sess.run([key, value])
    self.assertAllEqual(expected, k)
    self.assertAllEqual(expected, v)

  def testOneEpoch(self):
    with self.test_session() as sess:
      reader = tf.IdentityReader("test_reader")
      work_completed = reader.num_work_units_completed()
      produced = reader.num_records_produced()
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      queued_length = queue.size()
      key, value = reader.read(queue)

      self.assertAllEqual(0, work_completed.eval())
      self.assertAllEqual(0, produced.eval())
      self.assertAllEqual(0, queued_length.eval())

      queue.enqueue_many([["A", "B", "C"]]).run()
      queue.close().run()
      self.assertAllEqual(3, queued_length.eval())

      self._ExpectRead(sess, key, value, "A")
      self.assertAllEqual(1, produced.eval())

      self._ExpectRead(sess, key, value, "B")

      self._ExpectRead(sess, key, value, "C")
      self.assertAllEqual(3, produced.eval())
      self.assertAllEqual(0, queued_length.eval())

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])

      self.assertAllEqual(3, work_completed.eval())
      self.assertAllEqual(3, produced.eval())
      self.assertAllEqual(0, queued_length.eval())

  def testMultipleEpochs(self):
    with self.test_session() as sess:
      reader = tf.IdentityReader("test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      enqueue = queue.enqueue_many([["DD", "EE"]])
      key, value = reader.read(queue)

      enqueue.run()
      self._ExpectRead(sess, key, value, "DD")
      self._ExpectRead(sess, key, value, "EE")
      enqueue.run()
      self._ExpectRead(sess, key, value, "DD")
      self._ExpectRead(sess, key, value, "EE")
      enqueue.run()
      self._ExpectRead(sess, key, value, "DD")
      self._ExpectRead(sess, key, value, "EE")
      queue.close().run()
      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])

  def testSerializeRestore(self):
    with self.test_session() as sess:
      reader = tf.IdentityReader("test_reader")
      produced = reader.num_records_produced()
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      queue.enqueue_many([["X", "Y", "Z"]]).run()
      key, value = reader.read(queue)

      self._ExpectRead(sess, key, value, "X")
      self.assertAllEqual(1, produced.eval())
      state = reader.serialize_state().eval()

      self._ExpectRead(sess, key, value, "Y")
      self._ExpectRead(sess, key, value, "Z")
      self.assertAllEqual(3, produced.eval())

      queue.enqueue_many([["Y", "Z"]]).run()
      queue.close().run()
      reader.restore_state(state).run()
      self.assertAllEqual(1, produced.eval())
      self._ExpectRead(sess, key, value, "Y")
      self._ExpectRead(sess, key, value, "Z")
      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])
      self.assertAllEqual(3, produced.eval())

      self.assertEqual(str, type(state))

      with self.assertRaises(ValueError):
        reader.restore_state([])

      with self.assertRaises(ValueError):
        reader.restore_state([state, state])

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state(state[1:]).run()

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state(state[:-1]).run()

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state(state + "ExtraJunk").run()

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state("PREFIX" + state).run()

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state("BOGUS" + state[5:]).run()

  def testReset(self):
    with self.test_session() as sess:
      reader = tf.IdentityReader("test_reader")
      work_completed = reader.num_work_units_completed()
      produced = reader.num_records_produced()
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      queued_length = queue.size()
      key, value = reader.read(queue)

      queue.enqueue_many([["X", "Y", "Z"]]).run()
      self._ExpectRead(sess, key, value, "X")
      self.assertLess(0, queued_length.eval())
      self.assertAllEqual(1, produced.eval())

      self._ExpectRead(sess, key, value, "Y")
      self.assertLess(0, work_completed.eval())
      self.assertAllEqual(2, produced.eval())

      reader.reset().run()
      self.assertAllEqual(0, work_completed.eval())
      self.assertAllEqual(0, produced.eval())
      self.assertAllEqual(1, queued_length.eval())
      self._ExpectRead(sess, key, value, "Z")

      queue.enqueue_many([["K", "L"]]).run()
      self._ExpectRead(sess, key, value, "K")


class WholeFileReaderTest(tf.test.TestCase):

  def setUp(self):
    super(WholeFileReaderTest, self).setUp()
    self._filenames = [os.path.join(self.get_temp_dir(), "whole_file.%d.txt" % i)
                       for i in range(3)]
    self._content = ["One\na\nb\n", "Two\nC\nD", "Three x, y, z"]
    for fn, c in zip(self._filenames, self._content):
      open(fn, "w").write(c)

  def tearDown(self):
    super(WholeFileReaderTest, self).tearDown()
    for fn in self._filenames:
      os.remove(fn)

  def _ExpectRead(self, sess, key, value, index):
    k, v = sess.run([key, value])
    self.assertAllEqual(self._filenames[index], k)
    self.assertAllEqual(self._content[index], v)

  def testOneEpoch(self):
    with self.test_session() as sess:
      reader = tf.WholeFileReader("test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      queue.enqueue_many([self._filenames]).run()
      queue.close().run()
      key, value = reader.read(queue)

      self._ExpectRead(sess, key, value, 0)
      self._ExpectRead(sess, key, value, 1)
      self._ExpectRead(sess, key, value, 2)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])

  def testInfiniteEpochs(self):
    with self.test_session() as sess:
      reader = tf.WholeFileReader("test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      enqueue = queue.enqueue_many([self._filenames])
      key, value = reader.read(queue)

      enqueue.run()
      self._ExpectRead(sess, key, value, 0)
      self._ExpectRead(sess, key, value, 1)
      enqueue.run()
      self._ExpectRead(sess, key, value, 2)
      self._ExpectRead(sess, key, value, 0)
      self._ExpectRead(sess, key, value, 1)
      enqueue.run()
      self._ExpectRead(sess, key, value, 2)
      self._ExpectRead(sess, key, value, 0)


class TextLineReaderTest(tf.test.TestCase):

  def setUp(self):
    super(TextLineReaderTest, self).setUp()
    self._num_files = 2
    self._num_lines = 5

  def _LineText(self, f, l):
    return "%d: %d" % (f, l)

  def _CreateFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      f = open(fn, "w")
      for j in range(self._num_lines):
        f.write(self._LineText(i, j))
        # Always include a newline after the record unless it is
        # at the end of the file, in which case we include it sometimes.
        if j + 1 != self._num_lines or i == 0:
          f.write("\n")
    return filenames

  def testOneEpoch(self):
    files = self._CreateFiles()
    with self.test_session() as sess:
      reader = tf.TextLineReader(name="test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      key, value = reader.read(queue)

      queue.enqueue_many([files]).run()
      queue.close().run()
      for i in range(self._num_files):
        for j in range(self._num_lines):
          k, v = sess.run([key, value])
          self.assertAllEqual("%s:%d" % (files[i], j + 1), k)
          self.assertAllEqual(self._LineText(i, j), v)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        k, v = sess.run([key, value])

  def testSkipHeaderLines(self):
    files = self._CreateFiles()
    with self.test_session() as sess:
      reader = tf.TextLineReader(skip_header_lines=1, name="test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      key, value = reader.read(queue)

      queue.enqueue_many([files]).run()
      queue.close().run()
      for i in range(self._num_files):
        for j in range(self._num_lines - 1):
          k, v = sess.run([key, value])
          self.assertAllEqual("%s:%d" % (files[i], j + 2), k)
          self.assertAllEqual(self._LineText(i, j + 1), v)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        k, v = sess.run([key, value])


class FixedLengthRecordReaderTest(tf.test.TestCase):

  def setUp(self):
    super(FixedLengthRecordReaderTest, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._header_bytes = 5
    self._record_bytes = 3
    self._footer_bytes = 2

  def _Record(self, f, r):
    return str(f * 2 + r) * self._record_bytes

  def _CreateFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      f = open(fn, "w")
      f.write("H" * self._header_bytes)
      for j in range(self._num_records):
        f.write(self._Record(i, j))
      f.write("F" * self._footer_bytes)
    return filenames

  def testOneEpoch(self):
    files = self._CreateFiles()
    with self.test_session() as sess:
      reader = tf.FixedLengthRecordReader(
          header_bytes=self._header_bytes,
          record_bytes=self._record_bytes,
          footer_bytes=self._footer_bytes,
          name="test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      key, value = reader.read(queue)

      queue.enqueue_many([files]).run()
      queue.close().run()
      for i in range(self._num_files):
        for j in range(self._num_records):
          k, v = sess.run([key, value])
          self.assertAllEqual("%s:%d" % (files[i], j), k)
          self.assertAllEqual(self._Record(i, j), v)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        k, v = sess.run([key, value])


class TFRecordReaderTest(tf.test.TestCase):

  def setUp(self):
    super(TFRecordReaderTest, self).setUp()
    self._num_files = 2
    self._num_records = 7

  def _Record(self, f, r):
    return "Record %d of file %d" % (r, f)

  def _CreateFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = tf.python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        writer.write(self._Record(i, j))
    return filenames

  def testOneEpoch(self):
    files = self._CreateFiles()
    with self.test_session() as sess:
      reader = tf.TFRecordReader(name="test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      key, value = reader.read(queue)

      queue.enqueue_many([files]).run()
      queue.close().run()
      for i in range(self._num_files):
        for j in range(self._num_records):
          k, v = sess.run([key, value])
          self.assertTrue(k.startswith("%s:" % files[i]))
          self.assertAllEqual(self._Record(i, j), v)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        k, v = sess.run([key, value])


if __name__ == "__main__":
  tf.test.main()
