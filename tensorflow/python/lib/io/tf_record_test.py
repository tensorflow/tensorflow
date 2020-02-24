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
"""Tests for tf_record.TFRecordWriter and tf_record.tf_record_iterator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import random
import string
import zlib

import six

from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import test
from tensorflow.python.util import compat

TFRecordCompressionType = tf_record.TFRecordCompressionType

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
  """TFCompression Test"""

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


class TFRecordWriterTest(TFCompressionTestCase):
  """TFRecordWriter Test"""

  def _AssertFilesEqual(self, a, b, equal):
    for an, bn in zip(a, b):
      with open(an, "rb") as af, open(bn, "rb") as bf:
        if equal:
          self.assertEqual(af.read(), bf.read())
        else:
          self.assertNotEqual(af.read(), bf.read())

  def _CompressionSizeDelta(self, records, options_a, options_b):
    """Validate compression with options_a and options_b and return size delta.

    Compress records with options_a and options_b. Uncompress both compressed
    files and assert that the contents match the original records. Finally
    calculate how much smaller the file compressed with options_a was than the
    file compressed with options_b.

    Args:
      records: The records to compress
      options_a: First set of options to compress with, the baseline for size.
      options_b: Second set of options to compress with.

    Returns:
      The difference in file size when using options_a vs options_b. A positive
      value means options_a was a better compression than options_b. A negative
      value means options_b had better compression than options_a.

    """

    fn_a = self._WriteRecordsToFile(records, "tfrecord_a", options=options_a)
    test_a = list(tf_record.tf_record_iterator(fn_a, options=options_a))
    self.assertEqual(records, test_a, options_a)

    fn_b = self._WriteRecordsToFile(records, "tfrecord_b", options=options_b)
    test_b = list(tf_record.tf_record_iterator(fn_b, options=options_b))
    self.assertEqual(records, test_b, options_b)

    # Negative number => better compression.
    return os.path.getsize(fn_a) - os.path.getsize(fn_b)

  def testWriteReadZLibFiles(self):
    """test Write Read ZLib Files"""
    # Write uncompressed then compress manually.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
    files = self._CreateFiles(options, prefix="uncompressed")
    zlib_files = [
        self._ZlibCompressFile(fn, "tfrecord_%s.z" % i)
        for i, fn in enumerate(files)
    ]
    self._AssertFilesEqual(files, zlib_files, False)

    # Now write compressd and verify same.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    compressed_files = self._CreateFiles(options, prefix="compressed")
    self._AssertFilesEqual(compressed_files, zlib_files, True)

    # Decompress compress and verify same.
    uncompressed_files = [
        self._ZlibDecompressFile(fn, "tfrecord_%s.z" % i)
        for i, fn in enumerate(compressed_files)
    ]
    self._AssertFilesEqual(uncompressed_files, files, True)

  def testWriteReadGzipFiles(self):
    """test Write Read Gzip Files"""
    # Write uncompressed then compress manually.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
    files = self._CreateFiles(options, prefix="uncompressed")
    gzip_files = [
        self._GzipCompressFile(fn, "tfrecord_%s.gz" % i)
        for i, fn in enumerate(files)
    ]
    self._AssertFilesEqual(files, gzip_files, False)

    # Now write compressd and verify same.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    compressed_files = self._CreateFiles(options, prefix="compressed")

    # Note: Gzips written by TFRecordWriter add 'tfrecord_0' so
    # compressed_files can't be compared with gzip_files

    # Decompress compress and verify same.
    uncompressed_files = [
        self._GzipDecompressFile(fn, "tfrecord_%s.gz" % i)
        for i, fn in enumerate(compressed_files)
    ]
    self._AssertFilesEqual(uncompressed_files, files, True)

  def testNoCompressionType(self):
    """test No Compression Type"""
    self.assertEqual(
        "",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions()))

    self.assertEqual(
        "",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions("")))

    with self.assertRaises(ValueError):
      tf_record.TFRecordOptions(5)

    with self.assertRaises(ValueError):
      tf_record.TFRecordOptions("BZ2")

  def testZlibCompressionType(self):
    """test Zlib Compression Type"""
    zlib_t = tf_record.TFRecordCompressionType.ZLIB

    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions("ZLIB")))

    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions(zlib_t)))

    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions(tf_record.TFRecordOptions(zlib_t))))

  def testCompressionOptions(self):
    """Create record with mix of random and repeated data to test compression on."""
    rnd = random.Random(123)
    random_record = compat.as_bytes(
        "".join(rnd.choice(string.digits) for _ in range(10000)))
    repeated_record = compat.as_bytes(_TEXT)
    for _ in range(10000):
      start_i = rnd.randint(0, len(_TEXT))
      length = rnd.randint(10, 200)
      repeated_record += _TEXT[start_i:start_i + length]
    records = [random_record, repeated_record, random_record]

    tests = [
        ("compression_level", 2, -1),  # Lower compression is worse.
        ("compression_level", 6, 0),  # Default compression_level is equal.
        ("flush_mode", zlib.Z_FULL_FLUSH, 1),  # A few less bytes.
        ("flush_mode", zlib.Z_NO_FLUSH, 0),  # NO_FLUSH is the default.
        ("input_buffer_size", 4096, 0),  # Increases time not size.
        ("output_buffer_size", 4096, 0),  # Increases time not size.
        ("window_bits", 8, -1),  # Smaller than default window increases size.
        ("compression_strategy", zlib.Z_HUFFMAN_ONLY, -1),  # Worse.
        ("compression_strategy", zlib.Z_FILTERED, -1),  # Worse.
    ]

    compression_type = tf_record.TFRecordCompressionType.ZLIB
    options_a = tf_record.TFRecordOptions(compression_type)
    for prop, value, delta_sign in tests:
      options_b = tf_record.TFRecordOptions(
          compression_type=compression_type, **{prop: value})
      delta = self._CompressionSizeDelta(records, options_a, options_b)
      self.assertTrue(
          delta == 0 if delta_sign == 0 else delta // delta_sign > 0,
          "Setting {} = {}, file was {} smaller didn't match sign of {}".format(
              prop, value, delta, delta_sign))


class TFRecordWriterZlibTest(TFCompressionTestCase):
  """TFRecordWriter Zlib test"""

  def testZLibFlushRecord(self):
    """test ZLib Flush Record"""
    original = [b"small record"]
    fn = self._WriteRecordsToFile(original, "small_record")
    with open(fn, "rb") as h:
      buff = h.read()

    # creating more blocks and trailing blocks shouldn't break reads
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)

    output = b""
    for c in buff:
      if isinstance(c, int):
        c = six.int2byte(c)
      output += compressor.compress(c)
      output += compressor.flush(zlib.Z_FULL_FLUSH)

    output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FINISH)

    # overwrite the original file with the compressed data
    with open(fn, "wb") as h:
      h.write(output)

    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(fn, options=options))
    self.assertEqual(actual, original)

  def testZlibReadWrite(self):
    """Verify that files produced are zlib compatible."""
    original = [b"foo", b"bar"]
    fn = self._WriteRecordsToFile(original, "zlib_read_write.tfrecord")
    zfn = self._ZlibCompressFile(fn, "zlib_read_write.tfrecord.z")

    # read the compressed contents and verify.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(zfn, options=options))
    self.assertEqual(actual, original)

  def testZlibReadWriteLarge(self):
    """Verify that writing large contents also works."""

    # Make it large (about 5MB)
    original = [_TEXT * 10240]
    fn = self._WriteRecordsToFile(original, "zlib_read_write_large.tfrecord")
    zfn = self._ZlibCompressFile(fn, "zlib_read_write_large.tfrecord.z")

    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(zfn, options=options))
    self.assertEqual(actual, original)

  def testGzipReadWrite(self):
    """Verify that files produced are gzip compatible."""
    original = [b"foo", b"bar"]
    fn = self._WriteRecordsToFile(original, "gzip_read_write.tfrecord")
    gzfn = self._GzipCompressFile(fn, "tfrecord.gz")

    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    actual = list(tf_record.tf_record_iterator(gzfn, options=options))
    self.assertEqual(actual, original)


class TFRecordIteratorTest(TFCompressionTestCase):
  """TFRecordIterator test"""

  def setUp(self):
    super(TFRecordIteratorTest, self).setUp()
    self._num_records = 7

  def testIterator(self):
    """test Iterator"""
    records = [self._Record(0, i) for i in range(self._num_records)]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(records, "compressed_records", options)

    reader = tf_record.tf_record_iterator(fn, options)
    for expected in records:
      record = next(reader)
      self.assertEqual(expected, record)
    with self.assertRaises(StopIteration):
      record = next(reader)

  def testWriteZlibRead(self):
    """Verify compression with TFRecordWriter is zlib library compatible."""
    original = [b"foo", b"bar"]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(original, "write_zlib_read.tfrecord.z",
                                  options)

    zfn = self._ZlibDecompressFile(fn, "write_zlib_read.tfrecord")
    actual = list(tf_record.tf_record_iterator(zfn))
    self.assertEqual(actual, original)

  def testWriteZlibReadLarge(self):
    """Verify compression for large records is zlib library compatible."""
    # Make it large (about 5MB)
    original = [_TEXT * 10240]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(original, "write_zlib_read_large.tfrecord.z",
                                  options)
    zfn = self._ZlibDecompressFile(fn, "write_zlib_read_large.tfrecord")
    actual = list(tf_record.tf_record_iterator(zfn))
    self.assertEqual(actual, original)

  def testWriteGzipRead(self):
    original = [b"foo", b"bar"]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    fn = self._WriteRecordsToFile(original, "write_gzip_read.tfrecord.gz",
                                  options)

    gzfn = self._GzipDecompressFile(fn, "write_gzip_read.tfrecord")
    actual = list(tf_record.tf_record_iterator(gzfn))
    self.assertEqual(actual, original)

  def testReadGrowingFile_preservesReadOffset(self):
    """Verify that tf_record_iterator preserves read offset even after EOF.

    When a file is iterated to EOF, the iterator should raise StopIteration but
    not actually close the reader. Then if later new data is appended, the
    iterator should start returning that new data on the next call to next(),
    preserving the read offset. This behavior is required by TensorBoard.
    """
    # Start the file with a good record.
    fn = os.path.join(self.get_temp_dir(), "file.tfrecord")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"one")
      writer.write(b"two")
      writer.flush()
      iterator = tf_record.tf_record_iterator(fn)
      self.assertEqual(b"one", next(iterator))
      self.assertEqual(b"two", next(iterator))
      # Iterating at EOF results in StopIteration repeatedly.
      with self.assertRaises(StopIteration):
        next(iterator)
      with self.assertRaises(StopIteration):
        next(iterator)
      # Retrying after adding a new record successfully returns the new record,
      # preserving the prior read offset.
      writer.write(b"three")
      writer.flush()
      self.assertEqual(b"three", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)

  def testReadTruncatedFile_preservesReadOffset(self):
    """Verify that tf_record_iterator throws an exception on bad TFRecords.

    When a truncated record is completed, the iterator should return that new
    record on the next attempt at iteration, preserving the read offset. This
    behavior is required by TensorBoard.
    """
    # Write out a record and read it back it to get the raw bytes.
    fn = os.path.join(self.get_temp_dir(), "temp_file")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"truncated")
    with open(fn, "rb") as f:
      record_bytes = f.read()
    # Start the file with a good record.
    fn_truncated = os.path.join(self.get_temp_dir(), "truncated_file")
    with tf_record.TFRecordWriter(fn_truncated) as writer:
      writer.write(b"good")
    with open(fn_truncated, "ab", buffering=0) as f:
      # Cause truncation by omitting the last byte from the record.
      f.write(record_bytes[:-1])
      iterator = tf_record.tf_record_iterator(fn_truncated)
      # Good record appears first.
      self.assertEqual(b"good", next(iterator))
      # Truncated record repeatedly causes DataLossError upon iteration.
      with self.assertRaises(errors_impl.DataLossError):
        next(iterator)
      with self.assertRaises(errors_impl.DataLossError):
        next(iterator)
      # Retrying after completing the record successfully returns the rest of
      # the file contents, preserving the prior read offset.
      f.write(record_bytes[-1:])
      self.assertEqual(b"truncated", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)


class TFRecordRandomReaderTest(TFCompressionTestCase):

  def testRandomReaderReadingWorks(self):
    """Test read access to random offsets in the TFRecord file."""
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)

    offset = 0
    offsets = [offset]
    # Do a pass of forward reading.
    for i in range(self._num_records):
      record, offset = reader.read(offset)
      self.assertEqual(record, records[i])
      offsets.append(offset)
    # Reading off the bound should lead to error.
    with self.assertRaisesRegexp(IndexError, r"Out of range.*offset"):
      reader.read(offset)
    # Do a pass of backward reading.
    for i in range(self._num_records - 1, 0, -1):
      record, offset = reader.read(offsets[i])
      self.assertEqual(offset, offsets[i + 1])
      self.assertEqual(record, records[i])

  def testRandomReaderThrowsErrorForInvalidOffset(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    with self.assertRaisesRegexp(
        errors_impl.DataLossError, r"corrupted record"):
      reader.read(1)  # 1 is guaranteed to be an invalid offset.

  def testClosingRandomReaderCausesErrorsForFurtherReading(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    reader.close()
    with self.assertRaisesRegexp(
        errors_impl.FailedPreconditionError, r"closed"):
      reader.read(0)


class TFRecordWriterCloseAndFlushTests(test.TestCase):
  """TFRecordWriter close and flush tests"""

  # pylint: disable=arguments-differ
  def setUp(self, compression_type=TFRecordCompressionType.NONE):
    super(TFRecordWriterCloseAndFlushTests, self).setUp()
    self._fn = os.path.join(self.get_temp_dir(), "tf_record_writer_test.txt")
    self._options = tf_record.TFRecordOptions(compression_type)
    self._writer = tf_record.TFRecordWriter(self._fn, self._options)
    self._num_records = 20

  def _Record(self, r):
    return compat.as_bytes("Record %d" % r)

  def testWriteAndLeaveOpen(self):
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)

    # Verify no segfault if writer isn't explicitly closed.

  def testWriteAndRead(self):
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)
    self._writer.close()

    actual = list(tf_record.tf_record_iterator(self._fn, self._options))
    self.assertListEqual(actual, records)

  def testDoubleClose(self):
    self._writer.write(self._Record(0))
    self._writer.close()
    self._writer.close()

  def testFlushAfterCloseIsError(self):
    self._writer.write(self._Record(0))
    self._writer.close()

    with self.assertRaises(errors_impl.FailedPreconditionError):
      self._writer.flush()

  def testWriteAfterCloseIsError(self):
    self._writer.write(self._Record(0))
    self._writer.close()

    with self.assertRaises(errors_impl.FailedPreconditionError):
      self._writer.write(self._Record(1))


class TFRecordWriterCloseAndFlushGzipTests(TFRecordWriterCloseAndFlushTests):
  # pylint: disable=arguments-differ
  def setUp(self):
    super(TFRecordWriterCloseAndFlushGzipTests,
          self).setUp(TFRecordCompressionType.GZIP)


class TFRecordWriterCloseAndFlushZlibTests(TFRecordWriterCloseAndFlushTests):
  # pylint: disable=arguments-differ
  def setUp(self):
    super(TFRecordWriterCloseAndFlushZlibTests,
          self).setUp(TFRecordCompressionType.ZLIB)


if __name__ == "__main__":
  test.main()
