# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Dataset snapshot and related functionality."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


class _SnapshotDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A Dataset that captures a snapshot or reads from a snapshot."""

  def __init__(self, input_dataset, path):
    self._input_dataset = input_dataset
    self._path = ops.convert_to_tensor(path, dtype=dtypes.string, name="path")

    variant_tensor = ged_ops.snapshot_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        path=self._path,
        **dataset_ops.flat_structure(self))
    super(_SnapshotDataset, self).__init__(input_dataset, variant_tensor)


def snapshot(path):
  """Writes to/reads from a snapshot of a dataset.

  This function attempts to determine whether a valid snapshot exists at the
  `path`, and reads from the snapshot if so. If not, it will run the
  preprocessing pipeline as usual, and write out a snapshot of the data
  processed for future use.

  Args:
    path: A directory where we want to save our snapshots and/or read from a
      previously saved snapshot.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _SnapshotDataset(dataset, path)

  return _apply_fn
