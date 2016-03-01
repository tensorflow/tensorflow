# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for Reader ops from io_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import threading
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

      self._ExpectRead(sess, key, value, b"A")
      self.assertAllEqual(1, produced.eval())

      self._ExpectRead(sess, key, value, b"B")

      self._ExpectRead(sess, key, value, b"C")
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
      self._ExpectRead(sess, key, value, b"DD")
      self._ExpectRead(sess, key, value, b"EE")
      enqueue.run()
      self._ExpectRead(sess, key, value, b"DD")
      self._ExpectRead(sess, key, value, b"EE")
      enqueue.run()
      self._ExpectRead(sess, key, value, b"DD")
      self._ExpectRead(sess, key, value, b"EE")
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

      self._ExpectRead(sess, key, value, b"X")
      self.assertAllEqual(1, produced.eval())
      state = reader.serialize_state().eval()

      self._ExpectRead(sess, key, value, b"Y")
      self._ExpectRead(sess, key, value, b"Z")
      self.assertAllEqual(3, produced.eval())

      queue.enqueue_many([["Y", "Z"]]).run()
      queue.close().run()
      reader.restore_state(state).run()
      self.assertAllEqual(1, produced.eval())
      self._ExpectRead(sess, key, value, b"Y")
      self._ExpectRead(sess, key, value, b"Z")
      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        sess.run([key, value])
      self.assertAllEqual(3, produced.eval())

      self.assertEqual(bytes, type(state))

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
        reader.restore_state(state + b"ExtraJunk").run()

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state(b"PREFIX" + state).run()

      with self.assertRaisesOpError(
          "Could not parse state for IdentityReader 'test_reader'"):
        reader.restore_state(b"BOGUS" + state[5:]).run()

  def testReset(self):
    with self.test_session() as sess:
      reader = tf.IdentityReader("test_reader")
      work_completed = reader.num_work_units_completed()
      produced = reader.num_records_produced()
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      queued_length = queue.size()
      key, value = reader.read(queue)

      queue.enqueue_many([["X", "Y", "Z"]]).run()
      self._ExpectRead(sess, key, value, b"X")
      self.assertLess(0, queued_length.eval())
      self.assertAllEqual(1, produced.eval())

      self._ExpectRead(sess, key, value, b"Y")
      self.assertLess(0, work_completed.eval())
      self.assertAllEqual(2, produced.eval())

      reader.reset().run()
      self.assertAllEqual(0, work_completed.eval())
      self.assertAllEqual(0, produced.eval())
      self.assertAllEqual(1, queued_length.eval())
      self._ExpectRead(sess, key, value, b"Z")

      queue.enqueue_many([["K", "L"]]).run()
      self._ExpectRead(sess, key, value, b"K")


class WholeFileReaderTest(tf.test.TestCase):

  def setUp(self):
    super(WholeFileReaderTest, self).setUp()
    self._filenames = [os.path.join(self.get_temp_dir(), "whole_file.%d.txt" % i)
                       for i in range(3)]
    self._content = [b"One\na\nb\n", b"Two\nC\nD", b"Three x, y, z"]
    for fn, c in zip(self._filenames, self._content):
      open(fn, "wb").write(c)

  def tearDown(self):
    super(WholeFileReaderTest, self).tearDown()
    for fn in self._filenames:
      os.remove(fn)

  def _ExpectRead(self, sess, key, value, index):
    k, v = sess.run([key, value])
    self.assertAllEqual(tf.compat.as_bytes(self._filenames[index]), k)
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
    return tf.compat.as_bytes("%d: %d" % (f, l))

  def _CreateFiles(self, crlf=False):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      f = open(fn, "wb")
      for j in range(self._num_lines):
        f.write(self._LineText(i, j))
        # Always include a newline after the record unless it is
        # at the end of the file, in which case we include it sometimes.
        if j + 1 != self._num_lines or i == 0:
          f.write(b"\r\n" if crlf else b"\n")
    return filenames

  def _testOneEpoch(self, files):
    with self.test_session() as sess:
      reader = tf.TextLineReader(name="test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      key, value = reader.read(queue)

      queue.enqueue_many([files]).run()
      queue.close().run()
      for i in range(self._num_files):
        for j in range(self._num_lines):
          k, v = sess.run([key, value])
          self.assertAllEqual("%s:%d" % (files[i], j + 1), tf.compat.as_text(k))
          self.assertAllEqual(self._LineText(i, j), v)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        k, v = sess.run([key, value])

  def testOneEpochLF(self):
    self._testOneEpoch(self._CreateFiles(crlf=False))

  def testOneEpochCRLF(self):
    self._testOneEpoch(self._CreateFiles(crlf=True))

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
          self.assertAllEqual("%s:%d" % (files[i], j + 2), tf.compat.as_text(k))
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
    return tf.compat.as_bytes(str(f * 2 + r) * self._record_bytes)

  def _CreateFiles(self):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      f = open(fn, "wb")
      f.write(b"H" * self._header_bytes)
      for j in range(self._num_records):
        f.write(self._Record(i, j))
      f.write(b"F" * self._footer_bytes)
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
          self.assertAllEqual("%s:%d" % (files[i], j), tf.compat.as_text(k))
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
    return tf.compat.as_bytes("Record %d of file %d" % (r, f))

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
          self.assertTrue(tf.compat.as_text(k).startswith("%s:" % files[i]))
          self.assertAllEqual(self._Record(i, j), v)

      with self.assertRaisesOpError("is closed and has insufficient elements "
                                    "\\(requested 1, current size 0\\)"):
        k, v = sess.run([key, value])


class AsyncReaderTest(tf.test.TestCase):

  def testNoDeadlockFromQueue(self):
    """Tests that reading does not block main execution threads."""
    config = tf.ConfigProto(inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    with self.test_session(config=config) as sess:
      thread_data_t = collections.namedtuple("thread_data_t",
                                             ["thread", "queue", "output"])
      thread_data = []

      # Create different readers, each with its own queue.
      for i in range(3):
        queue = tf.FIFOQueue(99, [tf.string], shapes=())
        reader = tf.TextLineReader()
        _, line = reader.read(queue)
        output = []
        t = threading.Thread(target=AsyncReaderTest._RunSessionAndSave,
                             args=(sess, [line], output))
        thread_data.append(thread_data_t(t, queue, output))

      # Start all readers. They are all blocked waiting for queue entries.
      sess.run(tf.initialize_all_variables())
      for d in thread_data:
        d.thread.start()

      # Unblock the readers.
      for i, d in enumerate(reversed(thread_data)):
        fname = os.path.join(self.get_temp_dir(), "deadlock.%s.txt" % i)
        with open(fname, "wb") as f:
          f.write(("file-%s" % i).encode())
        d.queue.enqueue_many([[fname]]).run()
        d.thread.join()
        self.assertEqual([[("file-%s" % i).encode()]], d.output)

  @staticmethod
  def _RunSessionAndSave(sess, args, output):
    output.append(sess.run(args))


if __name__ == "__main__":
  tf.test.main()
