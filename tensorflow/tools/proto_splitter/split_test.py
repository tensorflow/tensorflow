# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Tests for ProtoSplitter."""

import os
import random
import string

import riegeli

from tensorflow.python.platform import test
from tensorflow.tools.proto_splitter import chunk_pb2
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter.testdata import test_message_pb2


class RepeatedStringSplitter(split.ComposableSplitter):
  """Splits a RepeatedString proto with N repeated strings into N chunks."""

  def __init__(self, proto, **kwargs):
    if not isinstance(proto, test_message_pb2.RepeatedString):
      raise TypeError("Can only split RepeatedString type protos")

    empty_strings = not bool(proto.strings)
    super().__init__(proto, proto_as_initial_chunk=empty_strings, **kwargs)

  def build_chunks(self):
    for n, s in enumerate(self._proto.strings):
      b = bytes(s, encoding="utf-8")
      self.add_chunk(b, ["strings", n])
    self._proto.ClearField("strings")


def _random_string(length):
  return bytes(
      "".join(random.choices(string.ascii_lowercase, k=length)),
      encoding="utf-8",
  )


class SplitRepeatedStringTest(test.TestCase):

  def _to_proto(self, strings):
    return test_message_pb2.RepeatedString(strings=strings)

  def testSplit(self):
    s = RepeatedStringSplitter(test_message_pb2.RepeatedString(strings=[]))
    chunks = s.split()[0]
    self.assertLen(chunks, 1)
    self.assertIsInstance(chunks[0], test_message_pb2.RepeatedString)

    s = RepeatedStringSplitter(
        test_message_pb2.RepeatedString(strings=["a", "b", "c"])
    )
    chunks, chunked_message = s.split()
    self.assertListEqual([b"a", b"b", b"c"], chunks)
    self.assertLen(chunked_message.chunked_fields, 3)

  def testWrite(self):
    path = os.path.join(self.create_tempdir(), "split-repeat")
    data = [_random_string(5), _random_string(10), _random_string(15)]
    returned_path = RepeatedStringSplitter(
        test_message_pb2.RepeatedString(strings=data)
    ).write(path)
    self.assertEqual(returned_path, f"{path}.cpb")

    with riegeli.RecordReader(open(f"{path}.cpb", "rb")) as reader:
      self.assertTrue(reader.check_file_format())
      records = list(reader.read_records())
      self.assertLen(records, 4)

      proto = chunk_pb2.ChunkMetadata()
      proto.ParseFromString(records[-1])
      self.assertLen(proto.message.chunked_fields, 3)
      self.assertFalse(proto.message.HasField("chunk_index"))

      expected_indices = [0, 1, 2]
      # Check that the chunk indices and info are correct.
      for expected_index, expected_data, chunk in zip(
          expected_indices, data, proto.message.chunked_fields
      ):
        i = chunk.message.chunk_index
        self.assertEqual(expected_index, i)

        chunk_info = proto.chunks[i]
        self.assertEqual(chunk_pb2.ChunkInfo.Type.BYTES, chunk_info.type)
        self.assertEqual(len(expected_data), chunk_info.size)
        reader.seek_numeric(chunk_info.offset)
        self.assertEqual(expected_data, reader.read_record())

  def test_child_splitter(self):
    proto = test_message_pb2.RepeatedRepeatedString(
        rs=[
            test_message_pb2.RepeatedString(strings=["a", "b", "c"]),
            test_message_pb2.RepeatedString(strings=["d", "e"]),
        ]
    )
    splitter = NoOpSplitter(proto)

    self.assertLen(splitter.split()[0], 1)
    splitter.add_chunk("", [])
    self.assertLen(splitter.split()[0], 2)

    child = RepeatedStringSplitter(
        proto.rs[0], parent_splitter=splitter, fields_in_parent=["rs", 0]
    )
    child.build_chunks()

    self.assertLen(splitter.split()[0], 5)  # Adds 3 chunks.

    RepeatedStringSplitter(
        proto.rs[1], parent_splitter=splitter, fields_in_parent=["rs", 1]
    ).build_chunks()

    self.assertLen(splitter.split()[0], 7)  # Adds 2 chunks.

    with self.assertRaisesRegex(
        ValueError, " `split` method should not be called directly"
    ):
      child.split()
    path = os.path.join(self.create_tempdir(), "na-split")
    with self.assertRaisesRegex(
        ValueError, " `write` method should not be called directly"
    ):
      child.write(path)


class NoOpSplitter(split.ComposableSplitter):

  def build_chunks(self):
    pass


class NoOpSplitterTest(test.TestCase):

  def testWriteNoChunks(self):
    path = os.path.join(self.create_tempdir(), "split-none")
    proto = test_message_pb2.RepeatedString(strings=["a", "bc", "de"])
    returned_path = NoOpSplitter(proto).write(path)

    expected_file_path = path + ".pb"
    self.assertTrue(os.path.isfile(expected_file_path))
    self.assertEqual(returned_path, expected_file_path)

    parsed_proto = test_message_pb2.RepeatedString()
    with open(expected_file_path, "rb") as f:
      parsed_proto.ParseFromString(f.read())
    self.assertProtoEquals(proto, parsed_proto)


if __name__ == "__main__":
  test.main()
