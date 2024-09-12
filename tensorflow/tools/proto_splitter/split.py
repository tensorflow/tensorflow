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
# ==============================================================================
"""Basic interface for Python-based Splitter."""

import abc
from collections.abc import Sequence
import time
from typing import Optional, Union

from absl import logging
import riegeli

from google.protobuf import message
from tensorflow.python.lib.io import file_io
from tensorflow.tools.proto_splitter import chunk_pb2
from tensorflow.tools.proto_splitter import util
from tensorflow.tools.proto_splitter import version as version_lib
from tensorflow.tools.proto_splitter import versions_pb2


class Splitter(abc.ABC):
  """An abstract class for splitting and writing protos that are > 2GB.

  See the README on how to use or subclass this class.
  """

  @property
  @abc.abstractmethod
  def version_def(self) -> versions_pb2.VersionDef:
    """Version info about the splitter and merge implementation required."""

  @abc.abstractmethod
  def split(
      self,
  ) -> tuple[Sequence[Union[message.Message, bytes]], chunk_pb2.ChunkedMessage]:
    """Splits proto message into a Sequence of protos/bytes."""

  @abc.abstractmethod
  def write(self, file_prefix: str) -> str:
    """Serializes proto to disk.

    Args:
      file_prefix: string prefix of the filepath.

    Returns:
      The actual path the proto is written to.
    """


class ComposableSplitter(Splitter):
  """A Splitter that can be composed with other splitters.

  This Splitter writes to the riegeli file format.

  See README for details.
  """

  def __init__(
      self,
      proto,
      *,
      proto_as_initial_chunk: bool = True,
      parent_splitter: Optional["ComposableSplitter"] = None,
      fields_in_parent: Optional[util.FieldTypes] = None,
  ):
    """Initializes ComposableSplitter.

    Args:
      proto: Proto message to split.
      proto_as_initial_chunk: Whether to initialize chunks with the
        user-provided proto as the initial chunk.
      parent_splitter: The parent `ComposableSplitter` object.
      fields_in_parent: Fields to access `proto` from the parent splitter's
        proto.
    """
    self._proto = proto
    self._parent_splitter = parent_splitter
    self._fields_in_parent = fields_in_parent

    # Whether chunks have been created. See `build_chunks()`.
    self._built = False

    # Keep a list of chunk ids in the order in which they were added to the
    # list.
    self._add_chunk_order = []
    self._fix_chunk_order = False

    # Initialize chunks and ChunkedMessage (optionally with the first chunk as
    # the user-provided proto.
    if parent_splitter is not None:
      # If this is not the root Splitter class, skip the initialization of
      # the chunks/message since the parent's will be updated instead.
      self._chunks = None
      self._chunked_message = None
    elif proto_as_initial_chunk:
      self._chunks = [self._proto]
      self._chunked_message = chunk_pb2.ChunkedMessage(chunk_index=0)
      self._add_chunk_order.append(id(self._proto))
    else:
      self._chunks = []
      self._chunked_message = chunk_pb2.ChunkedMessage()

  def build_chunks(self) -> None:
    """Builds the Splitter object by generating chunks from the proto.

    Subclasses of `ComposableChunks` should only need to override this method.

    This method should be called once per Splitter to create the chunks.
    Users should call the methods `split` or `write` instead.
    """

  @property
  def version_def(self) -> versions_pb2.VersionDef:
    """Version info about the splitter and join implementation required."""
    return versions_pb2.VersionDef(
        splitter_version=1,
        join_version=0,
        bad_consumers=version_lib.get_bad_versions(),
    )

  def split(
      self,
  ) -> tuple[Sequence[Union[message.Message, bytes]], chunk_pb2.ChunkedMessage]:
    """Splits a proto message into a Sequence of protos/bytes."""
    if self._parent_splitter:
      raise ValueError(
          "A child ComposableSplitter's `split` method should not be called "
          "directly, since it inherit chunks from a parent object. Please call "
          "the parent's `split()` method instead."
      )

    assert self._chunks is not None
    assert self._chunked_message is not None

    if not self._built:
      self.build_chunks()
      self._fix_chunks()
      self._built = True
    return self._chunks, self._chunked_message

  def write(
      self, file_prefix: str, writer_options: Optional[str] = None
  ) -> str:
    """Serializes a proto to disk.

    The writer writes all chunks into a riegeli file. The chunk metadata
    (ChunkMetadata) is written at the very end.

    Args:
      file_prefix: string prefix of the filepath. The writer will automatically
        attach a `.pb` or `.cpb` (chunked pb) suffix depending on whether the
        proto is split.
      writer_options: Optional writer options to pass to the riegeli writer. See
        https://github.com/google/riegeli/blob/master/doc/record_writer_options.md
        for options.

    Returns:
      The actual filepath the proto is written to. The filepath will be
      different depending on whether the proto is split, i.e., whether it will
      be a pb or not.
    """
    if self._parent_splitter is not None:
      raise ValueError(
          "A child ComposableSplitter's `write` method should not be called "
          "directly, since it inherits unrelated chunks from a parent object. "
          "Please call the parent's `write()` method instead."
      )

    start_time = time.time()
    chunks, chunked_message = self.split()

    if not chunked_message.chunked_fields:
      path = f"{file_prefix}.pb"
      file_io.atomic_write_string_to_file(
          path, self._proto.SerializeToString(deterministic=True)
      )
      logging.info("Unchunked file exported to %s", path)
      return path

    path = f"{file_prefix}.cpb"
    writer_kwargs = {}
    if writer_options is not None:
      writer_kwargs["options"] = writer_options
    with riegeli.RecordWriter(file_io.FileIO(path, "wb"), **writer_kwargs) as f:
      metadata = chunk_pb2.ChunkMetadata(
          message=chunked_message, version=self.version_def
      )
      for chunk in chunks:
        if isinstance(chunk, message.Message):
          f.write_message(chunk)
          chunk_type = chunk_pb2.ChunkInfo.Type.MESSAGE
          size = chunk.ByteSize()
        else:
          f.write_record(chunk)
          chunk_type = chunk_pb2.ChunkInfo.Type.BYTES
          size = len(chunk)
        metadata.chunks.add(
            type=chunk_type, size=size, offset=f.last_pos.numeric
        )
      f.write_message(metadata)

    end = time.time()

    logging.info("Chunked file exported to %s", path)
    logging.info(
        "Total time spent splitting and writing the message: %s",
        end - start_time,
    )
    logging.info(
        "Number of chunks created (including initial message): %s",
        len(chunks),
    )
    return path

  def add_chunk(
      self,
      chunk: Union[message.Message, bytes],
      field_tags: util.FieldTypes,
      index=None,
  ) -> None:
    """Adds a new chunk and updates the ChunkedMessage proto.

    Args:
      chunk: Proto message or bytes.
      field_tags: Field information about the placement of the chunked data
        within self._proto.
      index: Optional index at which to insert the chunk. The chunk ordering is
        important for merging.
    """
    if self._parent_splitter is not None:
      self._parent_splitter.add_chunk(
          chunk, self._fields_in_parent + field_tags, index
      )
    else:
      assert self._chunks is not None
      assert self._chunked_message is not None
      field = self._chunked_message.chunked_fields.add(
          field_tag=util.get_field_tag(self._proto, field_tags)
      )
      new_chunk_index = len(self._chunks)
      field.message.chunk_index = new_chunk_index
      self._add_chunk_order.append(id(chunk))

      if index is None:
        self._chunks.append(chunk)
      else:
        self._chunks.insert(index, chunk)
        self._fix_chunk_order = True

  def _fix_chunks(self) -> None:
    """Fixes chunk indices in the ChunkedMessage."""
    if not self._fix_chunk_order:
      return

    # The chunk_index of each nested ChunkedMessage is set to the length of the
    # list when the chunk was added. This would be fine if the chunks were
    # always added to the end of the list. However, this is not always the case
    # the indices must be updated.

    # Use the address of each chunk (python `id`) as lookup keys to the
    # ordered chunk indices.
    chunk_indices = {id(chunk): i for i, chunk in enumerate(self._chunks)}

    to_fix = [self._chunked_message]
    while to_fix:
      for field in to_fix.pop().chunked_fields:
        if field.message.chunked_fields:
          to_fix.append(field.message)
        if not field.message.HasField("chunk_index"):
          continue
        chunk_addr = self._add_chunk_order[field.message.chunk_index]
        assert (
            chunk_addr in chunk_indices
        ), f"Found unexpected chunk {chunk_addr}"
        new_chunk_index = chunk_indices[chunk_addr]
        field.message.chunk_index = new_chunk_index

    self._add_chunk_order = [id(chunk) for chunk in self._chunks]
    self._fix_chunk_order = False
