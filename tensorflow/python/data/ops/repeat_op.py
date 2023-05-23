# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.Dataset.repeat`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _repeat(input_dataset, count, name):  # pylint: disable=unused-private-name
  return _RepeatDataset(input_dataset, count, name)


class _RepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that repeats its input several times."""

  def __init__(self, input_dataset, count, name=None):
    """See `Dataset.repeat()` for details."""
    self._input_dataset = input_dataset
    if count is None:
      self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
    else:
      self._count = ops.convert_to_tensor(
          count, dtype=dtypes.int64, name="count")
    self._name = name
    variant_tensor = gen_dataset_ops.repeat_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
