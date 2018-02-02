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
"""Kafka Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kafka.python.ops import gen_kafka_ops
from tensorflow.contrib.util import loader
from tensorflow.python.data.ops.readers import Dataset
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader

class KafkaDataset(Dataset):
  """A Kafka Dataset that consumes the message.
  """

  def __init__(
      self, topics, servers="localhost", group="", eof=False, timeout=1000):
    """Create a KafkaReader.

    Args:
      topics: A `tf.string` tensor containing one or more subscriptions,
              in the format of [topic:partition:offset:length],
              by default length is -1 for unlimited.
      servers: A list of bootstrap servers.
      group: The consumer group id.
      eof: If True, the kafka reader will stop on EOF.
      timeout: The timeout value for the Kafka Consumer to wait
               (in millisecond).
    """
    super(KafkaDataset, self).__init__()
    self._topics = ops.convert_to_tensor(
        topics, dtype=dtypes.string, name="topics")
    self._servers = ops.convert_to_tensor(
        servers, dtype=dtypes.string, name="servers")
    self._group = ops.convert_to_tensor(
        group, dtype=dtypes.string, name="group")
    self._eof = ops.convert_to_tensor(
        eof, dtype=dtypes.bool, name="eof")
    self._timeout = ops.convert_to_tensor(
        timeout, dtype=dtypes.int64, name="timeout")

  def _as_variant_tensor(self):
    return gen_kafka_ops.kafka_dataset(
        self._topics, self._servers, self._group, self._eof, self._timeout)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string
