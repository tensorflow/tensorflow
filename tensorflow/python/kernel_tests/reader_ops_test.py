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
"""Tests for Reader ops from io_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import os
import shutil
import threading
import zlib

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.util import compat

prefix_path = "tensorflow/core/lib"

# pylint: disable=invalid-name
TFRecordCompressionType = tf_record.TFRecordCompressionType
# pylint: enable=invalid-name

# Edgar Allan Poe's 'Eldorado'
_TEXT = b"""Gaily bedight,
    A gallant knight,
    In sunshine and in shadow,
    Had journeyed long,
    Singing a song,
    In search of Eldorado.

    But he grew old
    This knight so bold
    And o'er his heart a shadow
    Fell as he found
    No spot of ground
    That looked like Eldorado.

   And, as his strength
   Failed him at length,
   He met a pilgrim shadow
   'Shadow,' said he,
   'Where can it be
   This land of Eldorado?'

   'Over the Mountains
    Of the Moon'
    Down the Valley of the Shadow,
    Ride, boldly ride,'
    The shade replied,
    'If you seek for Eldorado!'
    """


class TFCompressionTestCase(test.TestCase):

  def setUp(self):
    super(TFCompressionTestCase, self).setUp()
    self._num_files = 2
    self._num_records = 7

  def _Record(self, f, r):
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _CreateFiles(self, options=None, prefix=""):
    filenames = []
    for i in range(self._num_files):
      name = prefix + "tfrecord.%d.txt" % i
      records = [self._Record(i, j) for j in range(self._num_records)]
      fn = self._WriteRecordsToFile(records, name, options)
      filenames.append(fn)
    return filenames

  def _WriteRecordsToFile(self, records, name="tfrecord", options=None):
    fn = os.path.join(self.get_temp_dir(), name)
    with tf_record.TFRecordWriter(fn, options=options) as writer:
      for r in records:
        writer.write(r)
    return fn

  def _ZlibCompressFile(self, infile, name="tfrecord.z"):
    # zlib compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = zlib.compress(f.read())

    zfn = os.path.join(self.get_temp_dir(), name)
    with open(zfn, "wb") as f:
      f.write(cdata)
    return zfn

  def _GzipCompressFile(self, infile, name="tfrecord.gz"):
    # gzip compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = f.read()

    gzfn = os.path.join(self.get_temp_dir(), name)
    with gzip.GzipFile(gzfn, "wb") as f:
      f.write(cdata)
    return gzfn

  def _ZlibDecompressFile(self, infile, name="tfrecord"):
    with open(infile, "rb") as f:
      cdata = zlib.decompress(f.read())
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn

  def _GzipDecompressFile(self, infile, name="tfrecord"):
    with gzip.GzipFile(infile, "rb") as f:
      cdata = f.read()
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn


class IdentityReaderTest(test.TestCase):

  def _ExpectRead(self, key, value, expected):
    k, v = self.evaluate([key, value])
    self.assertAllEqual(expected, k)
    self.assertAllEqual(expected, v)

  def testOneEpoch(self):
    reader = io_ops.IdentityReader("test_reader")
    work_completed = reader.num_work_units_completed()
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    queued_length = queue.size()
    key, value = reader.read(queue)

    self.assertAllEqual(0, self.evaluate(work_completed))
    self.assertAllEqual(0, self.evaluate(produced))
    self.assertAllEqual(0, self.evaluate(queued_length))

    self.evaluate(queue.enqueue_many([["A", "B", "C"]]))
    self.evaluate(queue.close())
    self.assertAllEqual(3, self.evaluate(queued_length))

    self._ExpectRead(key, value, b"A")
    self.assertAllEqual(1, self.evaluate(produced))

    self._ExpectRead(key, value, b"B")

    self._ExpectRead(key, value, b"C")
    self.assertAllEqual(3, self.evaluate(produced))
    self.assertAllEqual(0, self.evaluate(queued_length))

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])

    self.assertAllEqual(3, self.evaluate(work_completed))
    self.assertAllEqual(3, self.evaluate(produced))
    self.assertAllEqual(0, self.evaluate(queued_length))

  def testMultipleEpochs(self):
    reader = io_ops.IdentityReader("test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    enqueue = queue.enqueue_many([["DD", "EE"]])
    key, value = reader.read(queue)

    self.evaluate(enqueue)
    self._ExpectRead(key, value, b"DD")
    self._ExpectRead(key, value, b"EE")
    self.evaluate(enqueue)
    self._ExpectRead(key, value, b"DD")
    self._ExpectRead(key, value, b"EE")
    self.evaluate(enqueue)
    self._ExpectRead(key, value, b"DD")
    self._ExpectRead(key, value, b"EE")
    self.evaluate(queue.close())
    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])

  def testSerializeRestore(self):
    reader = io_ops.IdentityReader("test_reader")
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    self.evaluate(queue.enqueue_many([["X", "Y", "Z"]]))
    key, value = reader.read(queue)

    self._ExpectRead(key, value, b"X")
    self.assertAllEqual(1, self.evaluate(produced))
    state = self.evaluate(reader.serialize_state())

    self._ExpectRead(key, value, b"Y")
    self._ExpectRead(key, value, b"Z")
    self.assertAllEqual(3, self.evaluate(produced))

    self.evaluate(queue.enqueue_many([["Y", "Z"]]))
    self.evaluate(queue.close())
    self.evaluate(reader.restore_state(state))
    self.assertAllEqual(1, self.evaluate(produced))
    self._ExpectRead(key, value, b"Y")
    self._ExpectRead(key, value, b"Z")
    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])
    self.assertAllEqual(3, self.evaluate(produced))

    self.assertEqual(bytes, type(state))

    with self.assertRaises(ValueError):
      reader.restore_state([])

    with self.assertRaises(ValueError):
      reader.restore_state([state, state])

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state[1:]))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state[:-1]))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state + b"ExtraJunk"))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(b"PREFIX" + state))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(b"BOGUS" + state[5:]))

  def testReset(self):
    reader = io_ops.IdentityReader("test_reader")
    work_completed = reader.num_work_units_completed()
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    queued_length = queue.size()
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([["X", "Y", "Z"]]))
    self._ExpectRead(key, value, b"X")
    self.assertLess(0, self.evaluate(queued_length))
    self.assertAllEqual(1, self.evaluate(produced))

    self._ExpectRead(key, value, b"Y")
    self.assertLess(0, self.evaluate(work_completed))
    self.assertAllEqual(2, self.evaluate(produced))

    self.evaluate(reader.reset())
    self.assertAllEqual(0, self.evaluate(work_completed))
    self.assertAllEqual(0, self.evaluate(produced))
    self.assertAllEqual(1, self.evaluate(queued_length))
    self._ExpectRead(key, value, b"Z")

    self.evaluate(queue.enqueue_many([["K", "L"]]))
    self._ExpectRead(key, value, b"K")


class WholeFileReaderTest(test.TestCase):

  def setUp(self):
    super(WholeFileReaderTest, self).setUp()
    self._filenames = [
        os.path.join(self.get_temp_dir(), "whole_file.%d.txt" % i)
        for i in range(3)
    ]
    self._content = [b"One\na\nb\n", b"Two\nC\nD", b"Three x, y, z"]
    for fn, c in zip(self._filenames, self._content):
      with open(fn, "wb") as h:
        h.write(c)

  def tearDown(self):
    for fn in self._filenames:
      os.remove(fn)
    super(WholeFileReaderTest, self).tearDown()

  def _ExpectRead(self, key, value, index):
    k, v = self.evaluate([key, value])
    self.assertAllEqual(compat.as_bytes(self._filenames[index]), k)
    self.assertAllEqual(self._content[index], v)

  def testOneEpoch(self):
    reader = io_ops.WholeFileReader("test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    self.evaluate(queue.enqueue_many([self._filenames]))
    self.evaluate(queue.close())
    key, value = reader.read(queue)

    self._ExpectRead(key, value, 0)
    self._ExpectRead(key, value, 1)
    self._ExpectRead(key, value, 2)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])

  def testInfiniteEpochs(self):
    reader = io_ops.WholeFileReader("test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    enqueue = queue.enqueue_many([self._filenames])
    key, value = reader.read(queue)

    self.evaluate(enqueue)
    self._ExpectRead(key, value, 0)
    self._ExpectRead(key, value, 1)
    self.evaluate(enqueue)
    self._ExpectRead(key, value, 2)
    self._ExpectRead(key, value, 0)
    self._ExpectRead(key, value, 1)
    self.evaluate(enqueue)
    self._ExpectRead(key, value, 2)
    self._ExpectRead(key, value, 0)


class TextLineReaderTest(test.TestCase):

  def setUp(self):
    super(TextLineReaderTest, self).setUp()
    self._num_files = 2
    self._num_lines = 5

  def _LineText(self, f, l):
    return compat.as_bytes("%d: %d" % (f, l))

  def _CreateFiles(self, crlf=False):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        for j in range(self._num_lines):
          f.write(self._LineText(i, j))
          # Always include a newline after the record unless it is
          # at the end of the file, in which case we include it sometimes.
          if j + 1 != self._num_lines or i == 0:
            f.write(b"\r\n" if crlf else b"\n")
    return filenames

  def _testOneEpoch(self, files):
    reader = io_ops.TextLineReader(name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_lines):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j + 1), compat.as_text(k))
        self.assertAllEqual(self._LineText(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def testOneEpochLF(self):
    self._testOneEpoch(self._CreateFiles(crlf=False))

  def testOneEpochCRLF(self):
    self._testOneEpoch(self._CreateFiles(crlf=True))

  def testSkipHeaderLines(self):
    files = self._CreateFiles()
    reader = io_ops.TextLineReader(skip_header_lines=1, name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_lines - 1):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j + 2), compat.as_text(k))
        self.assertAllEqual(self._LineText(i, j + 1), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])


class FixedLengthRecordReaderTest(TFCompressionTestCase):

  def setUp(self):
    super(FixedLengthRecordReaderTest, self).setUp()
    self._num_files = 2
    self._header_bytes = 5
    self._record_bytes = 3
    self._footer_bytes = 2

    self._hop_bytes = 2

  def _Record(self, f, r):
    return compat.as_bytes(str(f * 2 + r) * self._record_bytes)

  def _OverlappedRecord(self, f, r):
    record_str = "".join([
        str(i)[0]
        for i in range(r * self._hop_bytes,
                       r * self._hop_bytes + self._record_bytes)
    ])
    return compat.as_bytes(record_str)

  # gap_bytes=hop_bytes-record_bytes
  def _CreateFiles(self, num_records, gap_bytes):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_records > 0:
          f.write(self._Record(i, 0))
        for j in range(1, num_records):
          if gap_bytes > 0:
            f.write(b"G" * gap_bytes)
          f.write(self._Record(i, j))
        f.write(b"F" * self._footer_bytes)
    return filenames

  def _CreateOverlappedRecordFiles(self, num_overlapped_records):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(),
                        "fixed_length_overlapped_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_overlapped_records > 0:
          all_records_str = "".join([
              str(i)[0]
              for i in range(self._record_bytes + self._hop_bytes *
                             (num_overlapped_records - 1))
          ])
          f.write(compat.as_bytes(all_records_str))
        f.write(b"F" * self._footer_bytes)
    return filenames

  # gap_bytes=hop_bytes-record_bytes
  def _CreateGzipFiles(self, num_records, gap_bytes):
    filenames = self._CreateFiles(num_records, gap_bytes)
    for fn in filenames:
      # compress inplace.
      self._GzipCompressFile(fn, fn)
    return filenames

  # gap_bytes=hop_bytes-record_bytes
  def _CreateZlibFiles(self, num_records, gap_bytes):
    filenames = self._CreateFiles(num_records, gap_bytes)
    for fn in filenames:
      # compress inplace.
      self._ZlibCompressFile(fn, fn)
    return filenames

  def _CreateGzipOverlappedRecordFiles(self, num_overlapped_records):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(),
                        "fixed_length_overlapped_record.%d.txt" % i)
      filenames.append(fn)
      with gzip.GzipFile(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_overlapped_records > 0:
          all_records_str = "".join([
              str(i)[0]
              for i in range(self._record_bytes + self._hop_bytes *
                             (num_overlapped_records - 1))
          ])
          f.write(compat.as_bytes(all_records_str))
        f.write(b"F" * self._footer_bytes)
    return filenames

  def _CreateZlibOverlappedRecordFiles(self, num_overlapped_records):
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(),
                        "fixed_length_overlapped_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn + ".tmp", "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_overlapped_records > 0:
          all_records_str = "".join([
              str(i)[0]
              for i in range(self._record_bytes + self._hop_bytes *
                             (num_overlapped_records - 1))
          ])
          f.write(compat.as_bytes(all_records_str))
        f.write(b"F" * self._footer_bytes)
      self._ZlibCompressFile(fn + ".tmp", fn)
    return filenames

  # gap_bytes=hop_bytes-record_bytes
  def _TestOneEpoch(self, files, num_records, gap_bytes, encoding=None):
    hop_bytes = 0 if gap_bytes == 0 else self._record_bytes + gap_bytes
    reader = io_ops.FixedLengthRecordReader(
        header_bytes=self._header_bytes,
        record_bytes=self._record_bytes,
        footer_bytes=self._footer_bytes,
        hop_bytes=hop_bytes,
        encoding=encoding,
        name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(num_records):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j), compat.as_text(k))
        self.assertAllEqual(self._Record(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def _TestOneEpochWithHopBytes(self,
                                files,
                                num_overlapped_records,
                                encoding=None):
    reader = io_ops.FixedLengthRecordReader(
        header_bytes=self._header_bytes,
        record_bytes=self._record_bytes,
        footer_bytes=self._footer_bytes,
        hop_bytes=self._hop_bytes,
        encoding=encoding,
        name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(num_overlapped_records):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j), compat.as_text(k))
        self.assertAllEqual(self._OverlappedRecord(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def testOneEpoch(self):
    for num_records in [0, 7]:
      # gap_bytes=0: hop_bytes=0
      # gap_bytes=1: hop_bytes=record_bytes+1
      for gap_bytes in [0, 1]:
        files = self._CreateFiles(num_records, gap_bytes)
        self._TestOneEpoch(files, num_records, gap_bytes)

  def testGzipOneEpoch(self):
    for num_records in [0, 7]:
      # gap_bytes=0: hop_bytes=0
      # gap_bytes=1: hop_bytes=record_bytes+1
      for gap_bytes in [0, 1]:
        files = self._CreateGzipFiles(num_records, gap_bytes)
        self._TestOneEpoch(files, num_records, gap_bytes, encoding="GZIP")

  def testZlibOneEpoch(self):
    for num_records in [0, 7]:
      # gap_bytes=0: hop_bytes=0
      # gap_bytes=1: hop_bytes=record_bytes+1
      for gap_bytes in [0, 1]:
        files = self._CreateZlibFiles(num_records, gap_bytes)
        self._TestOneEpoch(files, num_records, gap_bytes, encoding="ZLIB")

  def testOneEpochWithHopBytes(self):
    for num_overlapped_records in [0, 2]:
      files = self._CreateOverlappedRecordFiles(num_overlapped_records)
      self._TestOneEpochWithHopBytes(files, num_overlapped_records)

  def testGzipOneEpochWithHopBytes(self):
    for num_overlapped_records in [0, 2]:
      files = self._CreateGzipOverlappedRecordFiles(num_overlapped_records,)
      self._TestOneEpochWithHopBytes(
          files, num_overlapped_records, encoding="GZIP")

  def testZlibOneEpochWithHopBytes(self):
    for num_overlapped_records in [0, 2]:
      files = self._CreateZlibOverlappedRecordFiles(num_overlapped_records)
      self._TestOneEpochWithHopBytes(
          files, num_overlapped_records, encoding="ZLIB")


class TFRecordReaderTest(TFCompressionTestCase):

  def setUp(self):
    super(TFRecordReaderTest, self).setUp()

  def testOneEpoch(self):
    files = self._CreateFiles()
    reader = io_ops.TFRecordReader(name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_records):
        k, v = self.evaluate([key, value])
        self.assertTrue(compat.as_text(k).startswith("%s:" % files[i]))
        self.assertAllEqual(self._Record(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def testReadUpTo(self):
    files = self._CreateFiles()
    reader = io_ops.TFRecordReader(name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    batch_size = 3
    key, value = reader.read_up_to(queue, batch_size)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    num_k = 0
    num_v = 0

    while True:
      try:
        k, v = self.evaluate([key, value])
        # Test reading *up to* batch_size records
        self.assertLessEqual(len(k), batch_size)
        self.assertLessEqual(len(v), batch_size)
        num_k += len(k)
        num_v += len(v)
      except errors_impl.OutOfRangeError:
        break

    # Test that we have read everything
    self.assertEqual(self._num_files * self._num_records, num_k)
    self.assertEqual(self._num_files * self._num_records, num_v)

  def testReadZlibFiles(self):
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    files = self._CreateFiles(options)

    reader = io_ops.TFRecordReader(name="test_reader", options=options)
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_records):
        k, v = self.evaluate([key, value])
        self.assertTrue(compat.as_text(k).startswith("%s:" % files[i]))
        self.assertAllEqual(self._Record(i, j), v)

  def testReadGzipFiles(self):
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    files = self._CreateFiles(options)

    reader = io_ops.TFRecordReader(name="test_reader", options=options)
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_records):
        k, v = self.evaluate([key, value])
        self.assertTrue(compat.as_text(k).startswith("%s:" % files[i]))
        self.assertAllEqual(self._Record(i, j), v)


class AsyncReaderTest(test.TestCase):

  def testNoDeadlockFromQueue(self):
    """Tests that reading does not block main execution threads."""
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    with self.session(config=config) as sess:
      thread_data_t = collections.namedtuple("thread_data_t",
                                             ["thread", "queue", "output"])
      thread_data = []

      # Create different readers, each with its own queue.
      for i in range(3):
        queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
        reader = io_ops.TextLineReader()
        _, line = reader.read(queue)
        output = []
        t = threading.Thread(
            target=AsyncReaderTest._RunSessionAndSave,
            args=(sess, [line], output))
        thread_data.append(thread_data_t(t, queue, output))

      # Start all readers. They are all blocked waiting for queue entries.
      self.evaluate(variables.global_variables_initializer())
      for d in thread_data:
        d.thread.start()

      # Unblock the readers.
      for i, d in enumerate(reversed(thread_data)):
        fname = os.path.join(self.get_temp_dir(), "deadlock.%s.txt" % i)
        with open(fname, "wb") as f:
          f.write(("file-%s" % i).encode())
        self.evaluate(d.queue.enqueue_many([[fname]]))
        d.thread.join()
        self.assertEqual([[("file-%s" % i).encode()]], d.output)

  @staticmethod
  def _RunSessionAndSave(sess, args, output):
    output.append(sess.run(args))


class LMDBReaderTest(test.TestCase):

  def setUp(self):
    super(LMDBReaderTest, self).setUp()
    # Copy database out because we need the path to be writable to use locks.
    path = os.path.join(prefix_path, "lmdb", "testdata", "data.mdb")
    self.db_path = os.path.join(self.get_temp_dir(), "data.mdb")
    shutil.copy(path, self.db_path)

  def testReadFromFile(self):
    reader = io_ops.LMDBReader(name="test_read_from_file")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue([self.db_path]))
    self.evaluate(queue.close())
    for i in range(10):
      k, v = self.evaluate([key, value])
      self.assertAllEqual(compat.as_bytes(k), compat.as_bytes(str(i)))
      self.assertAllEqual(
          compat.as_bytes(v), compat.as_bytes(str(chr(ord("a") + i))))

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def testReadFromSameFile(self):
    with self.cached_session() as sess:
      reader1 = io_ops.LMDBReader(name="test_read_from_same_file1")
      reader2 = io_ops.LMDBReader(name="test_read_from_same_file2")
      filename_queue = input_lib.string_input_producer(
          [self.db_path], num_epochs=None)
      key1, value1 = reader1.read(filename_queue)
      key2, value2 = reader2.read(filename_queue)

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)
      for _ in range(3):
        for _ in range(10):
          k1, v1, k2, v2 = self.evaluate([key1, value1, key2, value2])
          self.assertAllEqual(compat.as_bytes(k1), compat.as_bytes(k2))
          self.assertAllEqual(compat.as_bytes(v1), compat.as_bytes(v2))
      coord.request_stop()
      coord.join(threads)

  def testReadFromFolder(self):
    reader = io_ops.LMDBReader(name="test_read_from_folder")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue([self.db_path]))
    self.evaluate(queue.close())
    for i in range(10):
      k, v = self.evaluate([key, value])
      self.assertAllEqual(compat.as_bytes(k), compat.as_bytes(str(i)))
      self.assertAllEqual(
          compat.as_bytes(v), compat.as_bytes(str(chr(ord("a") + i))))

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def testReadFromFileRepeatedly(self):
    with self.cached_session() as sess:
      reader = io_ops.LMDBReader(name="test_read_from_file_repeated")
      filename_queue = input_lib.string_input_producer(
          [self.db_path], num_epochs=None)
      key, value = reader.read(filename_queue)

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)
      # Iterate over the lmdb 3 times.
      for _ in range(3):
        # Go over all 10 records each time.
        for j in range(10):
          k, v = self.evaluate([key, value])
          self.assertAllEqual(compat.as_bytes(k), compat.as_bytes(str(j)))
          self.assertAllEqual(
              compat.as_bytes(v), compat.as_bytes(str(chr(ord("a") + j))))
      coord.request_stop()
      coord.join(threads)


if __name__ == "__main__":
  test.main()
