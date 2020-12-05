# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Logic to read sharded files (tfrecord, buckets, ...).

This logic is shared between:
 - tfrecord_reader, to read sharded tfrecord files, based on user instructions.
 - tfrecord_writer, to read sharded bucket files (temp files), based on final
 sharding needs.
"""

from typing import List, Sequence

import attr


@attr.s(frozen=True)
class FileInstruction(object):  # TODO(epot): Uses dataclasses instead
  """Instruction to read a single shard/file.

  Attributes:
    filename: The filenames contains the relative path, not absolute.
    skip: Indicates which example read in the shard (`ds.skip().take()`). `None`
      if no skipping
    take: Indicates how many examples to read (`None` to read all)
    num_examples: `int`, The total number of examples
  """
  filename = attr.ib()
  skip = attr.ib()
  take = attr.ib()
  num_examples = attr.ib()

  def asdict(self):
    return {
        'filename': self.filename,
        'skip': self.skip,
        'take': self.take,
        'num_examples': self.num_examples,
    }

  def replace(self, **kwargs):
    new_attrs = self.asdict()
    new_attrs.update(kwargs)
    return type(self)(**new_attrs)


def get_file_instructions(
    from_: int,
    to: int,
    filenames: Sequence[str],
    shard_lengths: Sequence[int],
) -> List[FileInstruction]:
  """Returns a list of files (+skip/take) to read [from_:to] items from shards.

  Args:
    from_: int, Index (included) of element from which to read.
    to: int, Index (excluded) of element to which to read.
    filenames: list of strings or ints, the filenames of the shards. Not really
      used, but to place in result.
    shard_lengths: the number of elements in every shard.

  Returns:
    list of dict(filename, skip, take).
  """
  index_start = 0  # Beginning (included) of moving window.
  index_end = 0  # End (excluded) of moving window.
  file_instructions = []
  for filename, length in zip(filenames, shard_lengths):
    if not length:
      continue  # Empty shard - can happen with temporary buckets.
    index_end += length
    if from_ < index_end and to > index_start:  # There is something to take.
      skip = from_ - index_start if from_ > index_start else 0
      take = to - index_start - skip if to < index_end else -1
      if take == 0:
        continue
      file_instructions.append(FileInstruction(
          filename=filename,
          skip=skip,
          take=take,
          num_examples=length - skip if take == -1 else take,
      ))
    index_start += length
  return file_instructions
