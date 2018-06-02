# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Kinesis Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kinesis.python.ops import kinesis_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.kinesis.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class KinesisDataset(Dataset):
  """A Kinesis Dataset that consumes the message.
  """

  def __init__(self,
               stream,
               shard="",
               eof=False,
               interval=100000):
    """Create a KinesisReader.

    Args:
      stream: A `tf.string` tensor containing the name of the stream.
      shard: A `tf.string` tensor containing the id of the shard.
      eof: If True, the kinesis reader will stop on EOF.
      interval: The interval for the Kinesis Client to wait before
        try getting records again (in millisecond).
    """
    super(KinesisDataset, self).__init__()
    self._stream = ops.convert_to_tensor(
        stream, dtype=dtypes.string, name="stream")
    self._shard = ops.convert_to_tensor(
        shard, dtype=dtypes.string, name="shard")
    self._eof = ops.convert_to_tensor(eof, dtype=dtypes.bool, name="eof")
    self._interval = ops.convert_to_tensor(
        interval, dtype=dtypes.int64, name="interval")

  def _as_variant_tensor(self):
    return gen_dataset_ops.kinesis_dataset(
        self._stream, self._shard, self._eof, self._interval)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string
