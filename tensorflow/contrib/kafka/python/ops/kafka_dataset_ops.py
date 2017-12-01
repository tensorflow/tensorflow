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
"""Kafka Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kafka.python.ops import gen_kafka_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import resource_loader


class KafkaDataset(io_ops.ReaderBase):
  """A Kafka Reader that outputs the message.

  See ReaderBase for supported methods.
  """

  def __init__(self, servers="localhost", group=None, eof=False,
               timeout=1000, name=None):
    """Create a KafkaReader.

    Args:
      servers: A list of bootstrap servers.
      group: The consumer group id.
      eof: If True, the kafka dataset will stop on EOF.
      timeout: The timeout value for the Kafka Consumer to wait
               (in millisecond).
      name: A name for the operation (optional).
    """
    rr = gen_kafka_ops.kafka_dataset(servers=servers, group=group, eof=eof,
                                    timeout=timeout, name=name)
    super(KafkaDataset, self).__init__(rr)
