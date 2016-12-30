# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Masks one `Series` based on the content of another `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.contrib.learn.python.learn.dataframe.transforms import hashes


class SplitMask(transform.Transform):
  """Provide a boolean mask based on a hash of a `Series`."""

  def __init__(self, proportion):
    """Initialize `SplitMask`.

    Args:
      proportion: The proportion of the rows to select for the '1'
        partition; the remaining (1 - proportion) rows form the '0'
        partition.
    """
    # TODO(soergel): allow seed?
    super(SplitMask, self).__init__()
    self._proportion = proportion

  @property
  def name(self):
    return "SplitMask"

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output",

  def _produce_output_series(self, input_series=None):
    """Deterministically generate a boolean Series for partitioning rows.

    Note this split is only as deterministic as the underlying hash function;
    see `tf.string_to_hash_bucket_fast`.  The hash function is deterministic
    for a given binary, but may change occasionally.  The only way to achieve
    an absolute guarantee that the split `DataFrame`s do not change across runs
    is to materialize them.

    Note too that the allocation of a row to one partition or the
    other is evaluated independently for each row, so the exact number of rows
    in each partition is binomially distributed.

    Args:
      input_series: a `Series` of unique strings, whose hash will determine the
        partitioning.
        (This `Series` must contain strings because TensorFlow provides hash
        ops only for strings, and there are no number-to-string converter ops.)

    Returns:
      Two `DataFrame`s containing the partitioned rows.
    """
    # TODO(soergel): allow seed?
    num_buckets = 1000000  # close enough for simple splits
    hashed_input, = hashes.HashFast(num_buckets)(input_series[0])
    threshold = int(num_buckets * self._proportion)
    left_mask = hashed_input < threshold
    return [left_mask]

