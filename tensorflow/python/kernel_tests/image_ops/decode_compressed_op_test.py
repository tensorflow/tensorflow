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
"""Tests for DecodeRaw op from parsing_ops."""

import gzip
import io
import zlib

import zstandard as zstd

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class DecodeCompressedOpTest(test.TestCase):

  def _compress(self, bytes_in, compression_type):
    if not compression_type:
      return bytes_in
    elif compression_type == "ZLIB":
      return zlib.compress(bytes_in)
    elif compression_type == "ZSTD":
      return zstd.compress(bytes_in)
    else:
      out = io.BytesIO()
      with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(bytes_in)
      return out.getvalue()

  def testDecompressShapeInference(self):
    with ops.Graph().as_default():
      for compression_type in ["ZLIB", "GZIP", "ZSTD", ""]:
        with self.cached_session():
          in_bytes = array_ops.placeholder(dtypes.string, shape=[2])
          decompressed = parsing_ops.decode_compressed(
              in_bytes, compression_type=compression_type)
          self.assertEqual([2], decompressed.get_shape().as_list())

  def testDecompress(self):
    for compression_type in ["ZLIB", "GZIP", "ZSTD", ""]:
      with self.cached_session():

        def decode(in_bytes, compression_type=compression_type):
          return parsing_ops.decode_compressed(
              in_bytes, compression_type=compression_type)

        in_val = [self._compress(b"AaAA", compression_type),
                  self._compress(b"bBbb", compression_type)]
        result = self.evaluate(decode(in_val))
        self.assertAllEqual([b"AaAA", b"bBbb"], result)

  def testDecompressWithRaw(self):
    for compression_type in ["ZLIB", "GZIP", "ZSTD", ""]:
      with self.cached_session():

        def decode(in_bytes, compression_type=compression_type):
          decompressed = parsing_ops.decode_compressed(in_bytes,
                                                       compression_type)
          return parsing_ops.decode_raw(decompressed, out_type=dtypes.int16)

        result = self.evaluate(
            decode([self._compress(b"AaBC", compression_type)]))

        self.assertAllEqual(
            [[ord("A") + ord("a") * 256, ord("B") + ord("C") * 256]], result)

  def testDecompressZstdExceedsLimit(self):
    # Construct a ZSTD frame with uncompressed size 2GB (exceeding 1GB limit).
    # Magic Number: \x28\xb5\x2f\xfd
    # Frame_Header_Descriptor: \xc0
    # Window_Descriptor: \x13
    # Frame_Content_Size: \x00\x00\x00\x80\x00\x00\x00\x00 (2GB, 2**31)
    large_zstd_frame = (
        b"\x28\xb5\x2f\xfd\xc0\x13\x00\x00\x00\x80\x00\x00\x00\x00"
    )
    with self.cached_session():
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "exceeds maximum allowed size",
      ):
        self.evaluate(parsing_ops.decode_compressed([large_zstd_frame], "ZSTD"))


if __name__ == "__main__":
  test.main()
