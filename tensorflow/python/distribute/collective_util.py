# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for collectives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import tf_export


@tf_export("distribute.experimental.CollectiveHints")
class Hints(object):
  """Hints for collective operations like AllReduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Example:

  ```python
  hints = tf.distribute.experimental.CollectiveHints(
      bytes_per_pack=50 * 1024 * 1024)
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, experimental_hints=hints)
  optimizer.apply_gradients(zip(grads, vars), all_reduce_sum_gradients=False)
  ```

  """

  def __init__(self, bytes_per_pack=0):
    """Creates a CollectiveHints.

    Args:
      bytes_per_pack: A non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This only applies to all-reduce with
        `MultiWorkerMirroredStrategy` currently.

    Raises:
      ValueError: When arguments have invalid value.
    """
    if bytes_per_pack < 0:
      raise ValueError("bytes_per_pack must be non-negative")
    self.bytes_per_pack = bytes_per_pack
