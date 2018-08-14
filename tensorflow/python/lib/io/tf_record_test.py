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
import zlib

import six

from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import test
from tensorflow.python.util import compat

prefix_path = "third_party/tensorflow/core/lib"

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


class TFRecordWriterTest(TFCompressionTestCase):

  def setUp(self):
    super(TFRecordWriterTest, self).setUp()

  def _AssertFilesEqual(self, a, b, equal):
    for an, bn in zip(a, b):
      with open(an, "rb") as af, open(bn, "rb") as bf:
        if equal:
          self.assertEqual(af.read(), bf.read())
        else:
          self.assertNotEqual(af.read(), bf.read())

  def testWriteReadZLibFiles(self):
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


class TFRecordWriterZlibTest(TFCompressionTestCase):

  def testZLibFlushRecord(self):
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

  def setUp(self):
    super(TFRecordIteratorTest, self).setUp()
    self._num_records = 7

  def testIterator(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(records, "compressed_records", options)

    reader = tf_record.tf_record_iterator(fn, options)
    for expected in records:
      record = next(reader)
      self.assertAllEqual(expected, record)
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

  def testBadFile(self):
    """Verify that tf_record_iterator throws an exception on bad TFRecords."""
    fn = os.path.join(self.get_temp_dir(), "bad_file")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"123")
    fn_truncated = os.path.join(self.get_temp_dir(), "bad_file_truncated")
    with open(fn, "rb") as f:
      with open(fn_truncated, "wb") as f2:
        # DataLossError requires that we've written the header, so this must
        # be at least 12 bytes.
        f2.write(f.read(14))
    with self.assertRaises(errors_impl.DataLossError):
      for _ in tf_record.tf_record_iterator(fn_truncated):
        pass

class TFRecordWriterCloseAndFlushTests(test.TestCase):

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

  def testWriteAfterClose(self):
    self._writer.write(self._Record(0))
    self._writer.close()

    # TODO(sethtroisi): No way to know this failed, changed that.
    self._writer.write(self._Record(1))


class TFRecordWriterCloseAndFlushGzipTests(TFRecordWriterCloseAndFlushTests):

  def setUp(self):
    super(TFRecordWriterCloseAndFlushGzipTests,
          self).setUp(TFRecordCompressionType.GZIP)


class TFRecordWriterCloseAndFlushZlibTests(TFRecordWriterCloseAndFlushTests):

  def setUp(self):
    super(TFRecordWriterCloseAndFlushZlibTests,
          self).setUp(TFRecordCompressionType.ZLIB)


if __name__ == "__main__":
  test.main()
