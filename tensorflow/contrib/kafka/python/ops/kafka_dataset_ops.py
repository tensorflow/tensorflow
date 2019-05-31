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

from tensorflow.contrib.kafka.python.ops import gen_dataset_ops
from tensorflow.contrib.kafka.python.ops import kafka_op_loader  # pylint: disable=unused-import
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation


class KafkaDataset(dataset_ops.DatasetSource):
  """A Kafka Dataset that consumes the message.
  """

  @deprecation.deprecated(
      None,
      "tf.contrib.kafka will be removed in 2.0, the support for Apache Kafka "
      "will continue to be provided through the tensorflow/io GitHub project.")
  def __init__(self,
               topics,
               servers="localhost",
               group="",
               eof=False,
               timeout=1000):
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
    self._topics = ops.convert_to_tensor(
        topics, dtype=dtypes.string, name="topics")
    self._servers = ops.convert_to_tensor(
        servers, dtype=dtypes.string, name="servers")
    self._group = ops.convert_to_tensor(
        group, dtype=dtypes.string, name="group")
    self._eof = ops.convert_to_tensor(eof, dtype=dtypes.bool, name="eof")
    self._timeout = ops.convert_to_tensor(
        timeout, dtype=dtypes.int64, name="timeout")

    super(KafkaDataset, self).__init__(self._as_variant_tensor())

  def _as_variant_tensor(self):
    return gen_dataset_ops.kafka_dataset(self._topics, self._servers,
                                         self._group, self._eof, self._timeout)

  @property
  def _element_structure(self):
    return structure.TensorStructure(dtypes.string, [])
