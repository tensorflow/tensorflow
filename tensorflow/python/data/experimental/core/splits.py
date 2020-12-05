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

"""Splits related API."""


import typing
from typing import Any, List, Optional, Union

import dataclasses

from tensorflow.data.experimental.core import proto as proto_lib
from tensorflow.data.experimental.core import tfrecords_reader
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.utils import shard_utils
from tensorflow_metadata.proto.v0 import statistics_pb2


@dataclasses.dataclass(eq=False, frozen=True)
class SplitInfo:
  """Wraps `proto.SplitInfo` with an additional property.

  Attributes:
    name: Name of the split (e.g. `train`, `test`,...)
    shard_lengths: List of length <number of files> containing the number of
      examples stored in each file.
    num_examples: Total number of examples (`sum(shard_lengths)`)
    num_shards: Number of files (`len(shard_lengths)`)
    num_bytes: Size of the files
    statistics: Additional statistics of the split.
  """
  name: str
  shard_lengths: List[int]
  num_bytes: int
  statistics: statistics_pb2.DatasetFeatureStatistics = dataclasses.field(
      default_factory=statistics_pb2.DatasetFeatureStatistics,
  )
  # Inside `SplitDict`, `SplitInfo` has additional arguments required for
  # `file_instructions`
  # Rather than `dataset_name`, should use a structure containing file format,
  # data_dir,...
  _dataset_name: Optional[str] = None

  @classmethod
  def from_proto(cls, proto: proto_lib.SplitInfo) -> "SplitInfo":
    return cls(
        name=proto.name,
        shard_lengths=list(proto.shard_lengths),
        num_bytes=proto.num_bytes,
        statistics=proto.statistics,
    )

  def to_proto(self) -> proto_lib.SplitInfo:
    return proto_lib.SplitInfo(
        name=self.name,
        shard_lengths=self.shard_lengths,
        num_bytes=self.num_bytes,
        statistics=self.statistics if self.statistics.ByteSize() else None,
    )

  @property
  def num_examples(self) -> int:
    return sum(self.shard_lengths)

  @property
  def num_shards(self) -> int:
    return len(self.shard_lengths)

  def __repr__(self) -> str:
    num_examples = self.num_examples or "unknown"
    return f"<tfds.core.SplitInfo num_examples={num_examples}>"

  @property
  def file_instructions(self) -> List[shard_utils.FileInstruction]:
    """Returns the list of dict(filename, take, skip).

    This allows for creating your own `tf.data.Dataset` using the low-level
    TFDS values.

    Example:

    ```
    file_instructions = info.splits['train[75%:]'].file_instructions
    instruction_ds = tf.data.Dataset.from_generator(
        lambda: file_instructions,
        output_types={
            'filename': tf.string,
            'take': tf.int64,
            'skip': tf.int64,
        },
    )
    ds = instruction_ds.interleave(
        lambda f: tf.data.TFRecordDataset(
            f['filename']).skip(f['skip']).take(f['take'])
    )
    ```

    When `skip=0` and `take=-1`, the full shard will be read, so the `ds.skip`
    and `ds.take` could be skipped.

    Returns:
      A `dict(filename, take, skip)`
    """
    # `self._dataset_name` is assigned in `SplitDict.add()`.
    return tfrecords_reader.make_file_instructions(
        name=self._dataset_name,
        split_infos=[self],
        instruction=str(self.name),
    )

  @property
  def filenames(self) -> List[str]:
    """Returns the list of filenames."""
    return sorted(f.filename for f in self.file_instructions)

  def replace(self, **kwargs: Any) -> "SplitInfo":
    """Returns a copy of the `SplitInfo` with updated attributes."""
    return dataclasses.replace(self, **kwargs)


class SubSplitInfo(object):
  """Wrapper around a sub split info.

  This class expose info on the subsplit:

  ```
  ds, info = tfds.load(..., split='train[75%:]', with_info=True)
  info.splits['train[75%:]'].num_examples
  ```

  """

  def __init__(self, file_instructions: List[shard_utils.FileInstruction]):
    """Constructor.

    Args:
      file_instructions: List[FileInstruction]
    """
    self._file_instructions = file_instructions

  @property
  def num_examples(self) -> int:
    """Returns the number of example in the subsplit."""
    return sum(f.num_examples for f in self._file_instructions)

  @property
  def file_instructions(self) -> List[shard_utils.FileInstruction]:
    """Returns the list of dict(filename, take, skip)."""
    return self._file_instructions

  @property
  def filenames(self) -> List[str]:
    """Returns the list of filenames."""
    return sorted(f.filename for f in self.file_instructions)


# TODO(epot): `: tfds.Split` type should be `Union[str, Split]`
class Split(str):
  # pylint: disable=line-too-long
  """`Enum` for dataset splits.

  Datasets are typically split into different subsets to be used at various
  stages of training and evaluation.

  * `TRAIN`: the training data.
  * `VALIDATION`: the validation data. If present, this is typically used as
    evaluation data while iterating on a model (e.g. changing hyperparameters,
    model architecture, etc.).
  * `TEST`: the testing data. This is the data to report metrics on. Typically
    you do not want to use this during model iteration as you may overfit to it.

  See the
  [guide on splits](https://github.com/tensorflow/datasets/tree/master/docs/splits.md)
  for more information.
  """

  def __repr__(self) -> str:
    return "{}({})".format(type(self).__name__, super(Split, self).__repr__())  # pytype: disable=wrong-arg-types


Split.TRAIN = Split("train")
Split.TEST = Split("test")
Split.VALIDATION = Split("validation")

if typing.TYPE_CHECKING:
  # For type checking, `tfds.Split` is an alias for `str` with additional
  # `.TRAIN`, `.TEST`,... attributes. All strings are valid split type.
  Split = Union[Split, str]


class SplitDict(utils.NonMutableDict):
  """Split info object."""

  def __init__(self, split_infos: List[SplitInfo], *, dataset_name: str):
    # Forward the dataset name required to build file instructions:
    # info.splits['train'].file_instructions
    split_infos = [s.replace(_dataset_name=dataset_name) for s in split_infos]

    super(SplitDict, self).__init__(
        {split_info.name: split_info for split_info in split_infos},
        error_msg="Split {key} already present"
    )
    self._dataset_name = dataset_name

  def __getitem__(self, key):
    # 1st case: The key exists: `info.splits['train']`
    if str(key) in self:
      return super(SplitDict, self).__getitem__(str(key))
    # 2nd case: Uses instructions: `info.splits['train[50%]']`
    else:
      instructions = tfrecords_reader.make_file_instructions(
          name=self._dataset_name,
          split_infos=self.values(),
          instruction=key,
      )
      return SubSplitInfo(instructions)

  @classmethod
  def from_proto(cls, dataset_name, repeated_split_infos):
    """Returns a new SplitDict initialized from the `repeated_split_infos`."""
    split_infos = [SplitInfo.from_proto(s) for s in repeated_split_infos]
    return cls(split_infos, dataset_name=dataset_name)

  def to_proto(self):
    """Returns a list of SplitInfo protos that we have."""
    return [s.to_proto() for s in self.values()]

  @property
  def total_num_examples(self):
    """Return the total number of examples."""
    return sum(s.num_examples for s in self.values())


def even_splits(
    split: str,
    n: int,
) -> List[str]:
  """Generates a list of sub-splits of same size.

  Example:

  ```python
  assert tfds.even_splits('train', n=3) == [
      'train[0%:33%]', 'train[33%:67%]', 'train[67%:100%]',
  ]
  ```

  Args:
    split: Split name (e.g. 'train', 'test',...)
    n: Number of sub-splits to create

  Returns:
    The list of subsplits.
  """
  if n <= 0 or n > 100:
    raise ValueError(f"n should be > 0 and <= 100. Got {n}")
  partitions = [round(i * 100 / n) for i in range(n + 1)]
  return [
      f"{split}[{partitions[i]}%:{partitions[i+1]}%]" for i in range(n)
  ]
